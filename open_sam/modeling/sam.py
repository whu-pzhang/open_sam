from typing import Any, Dict, List, Tuple, Union
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

    @torch.no_grad()
    def get_image_embeddings(self, input_images):
        r"""
        Returns the image embeddings by passing the pixel values through the image encoder.

        Args:
            image (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
                Input pixel values
        """
        image_embeddings = self.image_encoder(input_images)[0]
        return image_embeddings

    def prompt_and_mask_decoder(self,
                                inputs,
                                image_embeddings,
                                multimask_output=False,
                                decoder_iter=False):
        if inputs["point_coords"] is not None:
            points = (inputs["point_coords"], inputs["point_labels"])
        else:
            points = None

        if decoder_iter:
            with torch.no_grad():
                sparse_embeddings, dense_embeddings = self.prompt_encoder(
                    points=points,
                    boxes=inputs.get("boxes", None),
                    masks=inputs.get("mask_inputs", None))

        else:
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=points,
                boxes=inputs.get("boxes", None),
                masks=inputs.get("mask_inputs", None))

        low_res_logits, iou_scores = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output)

        if multimask_output:
            b, _, h, w = low_res_logits.shape
            max_values, max_indexs = torch.max(iou_scores, dim=1, keepdim=True)
            iou_scores = max_values.unsqueeze(1)
            max_indexs = max_indexs.reshape(b, 1, 1, 1).expand(b, 1, h, w)
            low_res_logits = torch.gather(low_res_logits,
                                          dim=1,
                                          index=max_indexs)

        return low_res_logits, iou_scores

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
            return self._forward(inputs, multimask_output=multimask_output)
        else:
            raise RuntimeError(f'Invalid mode "{mode}". '
                               'Only supports loss, predict and tensor mode')

    def _stack_batch_gt(self, data_samples: SampleList) -> torch.Tensor:
        device = data_samples[0].gt_instances.bboxes.device
        gt_masks = [
            data_sample.gt_instances.masks.to_tensor(torch.long, device)
            for data_sample in data_samples
        ]
        gt_masks = torch.stack(gt_masks, dim=0)
        return gt_masks.reshape(-1, *gt_masks.shape[2:])

    def loss(self, batch_input, data_samples, multimask_output):
        gt_masks = self._stack_batch_gt(data_samples)

        multimask_output = True if random.random() > 0.5 else False
        low_res_logits, iou_scores = self._forward(
            batch_input, multimask_output=multimask_output)
        # handle multimask_output=True
        # if multimask_output:
        #     b, _, h, w = low_res_logits.shape
        #     max_values, max_indexs = torch.max(iou_scores, dim=1, keepdim=True)
        #     iou_scores = max_values.unsqueeze(1)
        #     max_indexs = max_indexs.reshape(b, 1, 1, 1).expand(b, 1, h, w)
        #     low_res_logits = torch.gather(low_res_logits,
        #                                   dim=1,
        #                                   index=max_indexs)

        if multimask_output:
            # During training, we compute the loss between the ground truth
            # and each of the predicted masks, but only backpropagate from
            # the lowest loss.
            num_masks = low_res_logits.size(1)
            loss_min = 9999
            for idx in range(num_masks):
                losses = self.loss_single(low_res_logits[:, idx:idx + 1],
                                          iou_scores[:, idx:idx + 1], gt_masks)
                loss_value = sum(losses.values())
                if loss_value < loss_min:
                    loss_dict = losses
                    loss_min = loss_value
        else:
            loss_dict = self.loss_single(low_res_logits, iou_scores, gt_masks)

        return loss_dict

    def loss_single(self, low_res_logits, iou_scores, gt_masks):
        logits = F.interpolate(low_res_logits,
                               size=gt_masks.shape[-2:],
                               mode='bilinear',
                               align_corners=False)
        with torch.no_grad():
            point_coords = get_uncertain_point_coords_with_randomness(
                logits, None, self.num_points, self.oversample_ratio,
                self.importance_sample_ratio)
            mask_point_targets = point_sample(
                gt_masks.unsqueeze(1).float(), point_coords).squeeze(1)
        mask_point_preds = point_sample(logits, point_coords)

        # cls loss
        loss_mask = self.loss_mask(mask_point_preds.reshape(-1),
                                   mask_point_targets.reshape(-1))
        # dice loss
        loss_dice = self.loss_dice(mask_point_preds, mask_point_targets)

        # loss_mask = self.loss_mask(logits, gt_masks.unsqueeze(1))
        # loss_dice = self.loss_dice(logits, gt_masks)
        batch_iou = calc_iou(logits.squeeze(1), gt_masks)
        loss_iou = self.loss_iou(iou_scores, batch_iou)

        return dict(loss_mask=loss_mask,
                    loss_dice=loss_dice,
                    loss_iou=loss_iou)

    def predict(self, batch_input, batch_data_samples, multimask_output=False):
        pred_masks, iou_predictions = self._forward(batch_input,
                                                    multimask_output)

        outputs = []
        for image_record, low_res_mask, iou_prediction in zip(
                batch_input, pred_masks, iou_predictions):
            masks = self.postprocess_masks(
                low_res_mask,
                input_size=image_record['image'].shape[-2:],
                original_size=image_record['original_size'],
            )

            masks = masks > self.mask_threshold
            outputs.append({
                'masks': masks,
                'iou_predictions': iou_prediction,
                'low_res_logits': low_res_mask,
            })

        # === TODO ===
        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, outputs)

        return batch_data_samples

    def add_pred_to_datasample(self, batch_data_samples, results_list):
        """Add predictions to `SamDataSample`.

        Args:
            data_samples (list[:obj:`SamDataSample`], optional): A batch of
                data samples that contain annotations and predictions.
            results_list (list[:obj:`dict`]): sam results of each image.
        """
        for data_sample, pred_output in zip(batch_data_samples, results_list):
            img_meta = data_sample.metainfo
            pred_masks = pred_output['masks']  # B,num_masks,H,W
            iou_scores = pred_output['iou_predictions']
            num_masks = pred_masks.size(1)

            if num_masks > 1:
                idx = torch.argmax(pred_output['iou_predictions'])
                iou_scores = iou_scores[:, idx]
                pred_masks = pred_masks[:, idx:idx + 1]
            pred_masks = pred_masks.squeeze(1)

            pred_instances = InstanceData()
            pred_instances.masks = pred_masks.detach()
            pred_instances.scores = iou_scores.detach()

            if data_sample.gt_instances:
                if data_sample.gt_instances.get('labels', None) is not None:
                    pred_instances.labels = data_sample.gt_instances['labels']
                else:
                    pred_instances.labels = [1] * len(pred_masks)
            else:
                pred_instances.labels = [None] * len(pred_masks)

            data_sample.pred_instances = pred_instances

        return batch_data_samples

    def _forward(self,
                 inputs: dict,
                 data_samples: OptSampleList = None,
                 multimask_output: bool = False):
        image_embeddings = self.image_encoder(inputs['image'])[0]

        num_prompts = inputs['boxes'].size(0) // image_embeddings.size(0)
        # Expand per-image data in batch direction to be per-mask
        image_embeddings = torch.repeat_interleave(image_embeddings,
                                                   num_prompts,
                                                   dim=0)

        if random.random() > 0.5:  # point prompt
            points = (inputs['point_coords'], inputs['point_labels'])
            inputs['boxes'] = None
        else:  # bbox prompt
            points = None

        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=points,
            boxes=inputs.get('boxes', None),
            masks=inputs.get('mask_inputs', None))

        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output)

        return low_res_masks, iou_predictions

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

    #TODO: support point iteration training
    def train_step(self, data: Union[dict, tuple, list],
                   optim_wrapper: OptimWrapper) -> Dict[str, torch.Tensor]:
        return super().train_step(data, optim_wrapper)
