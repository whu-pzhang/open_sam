from typing import Any, Dict, List, Tuple
import random
from mmengine.optim import OptimWrapper

import torch
from torch.nn import functional as F

from mmengine.model import BaseModel
from mmengine.structures import InstanceData
from mmseg.utils import OptConfigType, ConfigType
from mmdet.models.utils import get_uncertain_point_coords_with_randomness
from mmcv.ops import point_sample

from open_sam.registry import MODELS
from open_sam.datasets.sam_data_sample import SampleList, OptSampleList
from .utils import calc_iou


@MODELS.register_module()
class SAM(BaseModel):
    image_format: str = 'RGB'

    def __init__(self,
                 image_encoder: dict,
                 prompt_encoder: dict,
                 mask_decoder: dict,
                 mask_threshold: float = 0.,
                 loss_mask: ConfigType = dict(type='mmdet.CrossEntropyLoss',
                                              use_sigmoid=True,
                                              reduction='mean',
                                              loss_weight=5.0),
                 loss_dice: ConfigType = dict(type='mmdet.DiceLoss',
                                              use_sigmoid=True,
                                              activate=True,
                                              reduction='mean',
                                              naive_dice=True,
                                              eps=1.0,
                                              loss_weight=5.0),
                 loss_iou: ConfigType = dict(type='mmdet.MSELoss',
                                             loss_weight=1.0),
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptConfigType = None) -> None:
        """SAM predicts object masks from an image and input prompts. Borrowed
        from https://github.com/facebookresearch/segment-anything.

        Arguments:
          image_encoder (ViTSAM): The backbone used to encode the
            image into image embeddings that allow for efficient mask
            prediction.
          prompt_encoder (PromptEncoder): Encodes various types of input
            prompts.
          mask_decoder (MaskDecoder): Predicts masks from the image embeddings
            and encoded prompts.
          pixel_mean (list(float)): Mean values for normalizing pixels in the
            input image.
          pixel_std (list(float)): Std values for normalizing pixels in the
            input image.
        """
        super().__init__(data_preprocessor=data_preprocessor,
                         init_cfg=init_cfg)
        self.image_encoder = MODELS.build(image_encoder)
        self.prompt_encoder = MODELS.build(prompt_encoder)
        self.mask_decoder = MODELS.build(mask_decoder)

        self.mask_threshold = mask_threshold
        self.ignore_index = 255

        self.register_buffer(
            'pixel_mean',
            torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1), False)
        self.register_buffer(
            'pixel_std',
            torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1), False)

        # =========
        self.test_cfg = test_cfg
        self.train_cfg = train_cfg
        if train_cfg:
            self.num_points = self.train_cfg.get('num_points', 12544)
            self.oversample_ratio = self.train_cfg.get('oversample_ratio', 3.0)
            self.importance_sample_ratio = self.train_cfg.get(
                'importance_sample_ratio', 0.75)

        self.loss_mask = MODELS.build(loss_mask)
        self.loss_dice = MODELS.build(loss_dice)
        self.loss_iou = MODELS.build(loss_iou)

    def init_weights(self):
        #TODO: positional embedding resize
        super().init_weights()

    @property
    def device(self) -> Any:
        return self.pixel_mean.device

    def prompt_and_mask_decoder_forward(self,
                                        inputs,
                                        image_embeddings,
                                        multimask_output=True,
                                        decoder_iter=False):
        logits_low_res = []
        iou_scores = []

        for idx, curr_embedding in enumerate(image_embeddings):
            if inputs['point_coords'][idx] is not None:
                points = (inputs['point_coords'][idx],
                          inputs['point_labels'][idx])
            else:
                points = None

            if decoder_iter:
                with torch.no_grad():
                    sparse_embeddings, dense_embeddings = self.prompt_encoder(
                        points=points,
                        boxes=inputs['boxes'][idx],
                        masks=inputs['masks'][idx])
            else:
                sparse_embeddings, dense_embeddings = self.prompt_encoder(
                    points=points,
                    boxes=inputs['boxes'][idx],
                    masks=inputs['masks'][idx])

            low_res_masks, iou_predictions = self.mask_decoder(
                image_embeddings=curr_embedding.unsqueeze(0),
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output)

            logits_low_res.append(low_res_masks)
            iou_scores.append(iou_predictions)

        return logits_low_res, iou_scores

    def forward(self,
                inputs,
                data_samples=None,
                mode='loss',
                multimask_output=False):
        '''
        Args:
            inputs (torch.Tensor): The input tensor with shape (N, C, ...) in
                general.
            data_samples (list[:obj:`SamDataSample`]): The sam data samples.
                It usually includes information such as `metainfo` and
                `gt_masks`. Default to None.
            mode (str): Return what kind of value. Defaults to 'tensor'.

        Returns:
            The return type depends on ``mode``.

            - If ``mode="tensor"``, return a tensor or a tuple of tensor.
            - If ``mode="predict"``, return a list of :obj:`SamDataSample`.
            - If ``mode="loss"``, return a dict of tensor.
        '''
        if mode == 'loss':
            return self.loss(inputs, data_samples, multimask_output=True)
        elif mode == 'predict':
            return self.predict(inputs, data_samples, multimask_output)
        elif mode == 'tensor':
            return self._forward(inputs, multimask_output)
        else:
            raise RuntimeError(f'Invalid mode "{mode}". '
                               'Only supports loss, predict and tensor mode')

    def _stack_batch_gt(self, data_samples: SampleList) -> torch.Tensor:
        device = data_samples[0].gt_instances.bboxes.device
        gt_masks = [
            data_sample.gt_instances.masks.to_tensor(torch.long, device)
            for data_sample in data_samples
        ]
        return gt_masks

    def loss(self,
             inputs: dict,
             data_samples: OptSampleList = None,
             multimask_output: bool = False):
        gt_masks = self._stack_batch_gt(data_samples)

        low_res_logits, iou_predictions = self._forward(
            inputs, multimask_output=multimask_output)

        pred_logits = []
        loss_dict = dict(loss_mask=0., loss_dice=0., loss_iou=0.)
        for batch_idx, (logits, iou_scores) in enumerate(
                zip(low_res_logits, iou_predictions)):
            # 10,3,256,256; 10,3; 10,1024,1024
            gt_mask = gt_masks[batch_idx]
            logits = F.interpolate(logits,
                                   size=gt_mask.shape[-2:],
                                   mode='bilinear',
                                   align_corners=False)
            pred_logits.append(logits)

            # handle multiple-mask output
            num_masks_per_prompt = logits.size(1)
            num_masks = gt_mask.size(0)

            loss_min = float('inf')
            cur_loss_dict = dict()
            for idx in range(num_masks_per_prompt):
                with torch.no_grad():
                    point_coords = get_uncertain_point_coords_with_randomness(
                        logits[:, idx:idx + 1], None, self.num_points,
                        self.oversample_ratio, self.importance_sample_ratio)
                    mask_point_targets = point_sample(
                        gt_mask.unsqueeze(1).float(), point_coords).squeeze(1)

                mask_point_preds = point_sample(logits[:, idx:idx + 1],
                                                point_coords)

                # cls loss
                loss_mask = self.loss_mask(mask_point_preds.reshape(-1),
                                           mask_point_targets.reshape(-1),
                                           avg_factor=num_masks *
                                           self.num_points)
                # dice loss
                loss_dice = self.loss_dice(mask_point_preds,
                                           mask_point_targets,
                                           avg_factor=num_masks)
                # iou loss
                batch_iou = calc_iou(logits[:, idx], gt_mask)
                loss_iou = self.loss_iou(iou_scores[:, idx:idx + 1],
                                         batch_iou,
                                         avg_factor=num_masks)

                loss_value = loss_mask + loss_dice + loss_iou
                if loss_value < loss_min:
                    cur_loss_dict.update(loss_mask=loss_mask,
                                         loss_dice=loss_dice,
                                         loss_iou=loss_iou)
                    loss_min = loss_value

            for k, v in cur_loss_dict.items():
                loss_dict[k] += v

        batch_size = len(data_samples)
        loss_dict = {k: v / batch_size for k, v in loss_dict.items()}

        return loss_dict, pred_logits

    def predict(self,
                inputs,
                data_samples: OptSampleList,
                multimask_output: bool = False):
        batch_img_metas = [
            data_sample.metainfo for data_sample in data_samples
        ]
        pred_logits, iou_predictions = self._forward(
            inputs, multimask_output=multimask_output)

        outputs = []
        for img_id in range(len(batch_img_metas)):
            img_meta = batch_img_metas[img_id]
            logits = pred_logits[img_id]
            iou_scores = iou_predictions[img_id]

            logits_pred = F.interpolate(logits,
                                        self.image_encoder.img_size,
                                        mode='bilinear',
                                        align_corners=False)
            img_shape = img_meta['img_shape']
            logits_pred = logits_pred[..., :img_shape[0], :img_shape[1]]
            logits_pred = F.interpolate(logits_pred,
                                        size=img_meta['ori_shape'],
                                        mode='bilinear',
                                        align_corners=False)
            masks_pred = logits_pred > self.mask_threshold

            outputs.append({
                'masks': masks_pred,
                'iou_predictions': iou_scores,
                'low_res_logits': logits
            })

        data_samples = self.add_pred_to_datasample(data_samples, outputs)

        return data_samples

    def add_pred_to_datasample(self, batch_data_samples, results_list):
        """Add predictions to `SamDataSample`.

        Args:
            data_samples (list[:obj:`SamDataSample`], optional): A batch of
                data samples that contain annotations and predictions.
            results_list (list[:obj:`dict`]): sam results of each image.
        """
        for data_sample, pred_output in zip(batch_data_samples, results_list):
            pred_masks = pred_output['masks']  # B,num_masks,H,W
            iou_scores = pred_output['iou_predictions']
            num_masks = pred_masks.size(1)

            if num_masks > 1:
                iou_scores, idx = torch.max(pred_output['iou_predictions'])
                pred_masks = pred_masks[:, idx:idx + 1]
            pred_masks = pred_masks.squeeze(1)

            pred_instances = InstanceData()
            pred_instances.masks = pred_masks.detach()
            pred_instances.scores = iou_scores.detach()

            data_sample.pred_instances = pred_instances

        return batch_data_samples

    def _forward(
            self,
            inputs: dict,
            data_samples: OptSampleList = None,
            multimask_output: bool = False) -> List[Dict[str, torch.Tensor]]:
        """Predicts masks end-to-end from provided images and prompts. If
        prompts are not known in advance, using SamPredictor is recommended
        over calling the model directly.

        Borrowed from https://github.com/facebookresearch/segment-anything

        Arguments:
          inputs (dict):  a dictionary with the following keys
              'image': The image as a torch tensor in Bx3xHxW format,
                already transformed for input to the model.
              'original_size': (tuple(int, int)) The original size of
                the image before transformation, as (H, W).
              'point_coords': (list[torch.Tensor]) Batched point prompts for
                each image, each with shape BxNx2. Already transformed to the
                input frame of the model.
              'point_labels': (list[torch.Tensor]) Batched labels for point prompts,
                each tensor with shape BxN.
              'boxes': (list[torch.Tensor]) Batched box inputs, each with shape Bx4.
                Already transformed to the input frame of the model.
              'mask_inputs': (list[torch.Tensor]) Batched mask inputs to the model,
                each mask in the form Bx1xHxW.
          multimask_output (bool): Whether the model should predict multiple
            disambiguating masks, or return a single mask.

        Returns:
          (list(dict)): A list over input images, where each element is
            as dictionary with the following keys.
              'masks': (torch.Tensor) Batched binary mask predictions,
                with shape BxCxHxW, where B is the number of input prompts,
                C is determiend by multimask_output, and (H, W) is the
                original size of the image.
              'iou_predictions': (torch.Tensor) The model's predictions
                of mask quality, in shape BxC.
              'low_res_logits': (torch.Tensor) Low resolution logits with
                shape BxCxHxW, where H=W=256. Can be passed as mask input
                to subsequent iterations of prediction.
        """
        image_embeddings = self.image_encoder(inputs['image'])[0]
        del inputs['image']

        # format inputs
        for idx in range(image_embeddings.size(0)):
            prompt_type = random.choice(inputs['prompt_type'][idx])
            if prompt_type == 'point':
                inputs['boxes'][idx] = None
            elif prompt_type == 'bbox':
                inputs['point_coords'][idx] = None
                inputs['point_labels'][idx] = None

        logits_low_res, iou_scores = self.prompt_and_mask_decoder_forward(
            inputs, image_embeddings, multimask_output, decoder_iter=False)

        return logits_low_res, iou_scores

    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """Remove padding and upscale masks to the original image size.

        Borrowed from https://github.com/facebookresearch/segment-anything

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        masks = F.interpolate(masks,
                              self.image_encoder.img_size,
                              mode='bilinear',
                              align_corners=False)
        masks = masks[..., :input_size[0], :input_size[1]]
        masks = F.interpolate(masks,
                              original_size,
                              mode='bilinear',
                              align_corners=False)
        return masks

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        img_size = max(self.image_encoder.img_size)
        padh = img_size - h
        padw = img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def train_step(self, data,
                   optim_wrapper: OptimWrapper) -> Dict[str, torch.Tensor]:
        with optim_wrapper.optim_context(self):
            data = self.data_preprocessor(data, True)
            losses_main, pred_logits = self.forward(**data, mode='loss')

        parsed_losses, log_vars = self.parse_losses(losses_main)

        # 1. first update
        # update_params:
        #   1. loss.backward()
        #   2. step
        #   3. zero_grad()
        optim_wrapper.update_params(parsed_losses)

        # 2. second update
        #

        return log_vars
