from typing import Any, Dict, List, Tuple
import random

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

    def prompt_and_mask_decoder(self):
        pass

    def _format_inputs(self, inputs, data_samples):
        '''
        batched_inputs: List[dict]  

        Returns:
            - image
            - original_size
            - point_coords
            - point_labels
            - boxes
        '''
        batched_inputs = []
        for idx, (img, data_sample) in enumerate(zip(inputs, data_samples)):
            metainfo = data_sample.metainfo
            prompt_type = metainfo['prompt_type']
            prompt_instances = data_samples[idx].prompt_instances

            inputs = dict(image=img, original_size=metainfo['original_size'])

            if prompt_type == 'point':
                inputs.update(point_coords=prompt_instances.point_coords,
                              point_labels=prompt_instances.point_labels)
            elif prompt_type == 'boxes':
                inputs.update(boxes=prompt_instances.boxes)

            batched_inputs.append(inputs)

        return batched_inputs

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
            return self.loss(inputs, data_samples, multimask_output)
        elif mode == 'predict':
            return self.predict(inputs, data_samples, multimask_output)
        elif mode == 'tensor':
            return self._forward(inputs, multimask_output=multimask_output)
        else:
            raise RuntimeError(f'Invalid mode "{mode}". '
                               'Only supports loss, predict and tensor mode')

    def loss(self, batch_input, data_samples, multimask_output=False):
        low_res_logits, iou_predictions = self._forward(
            batch_input, multimask_output=multimask_output)
        print(low_res_logits.shape, iou_predictions.shape)

        loss_dict = dict()
        batch_size = len(data_samples)
        loss_mask = 0
        loss_dice = 0
        loss_iou = 0

        for batch_idx, (logits, iou_scores) in enumerate(
                zip(low_res_logits, iou_predictions)):

            gt_masks = data_samples[batch_idx].gt_instances.masks
            high_res_logits = F.interpolate(logits,
                                            size=gt_masks.shape[-2:],
                                            mode='bilinear',
                                            align_corners=False)

            # handle multiple-mask output
            #
            num_masks_per_prompt = high_res_logits.size(1)
            if num_masks_per_prompt > 1:
                b, num_masks, h, w = high_res_logits.shape
                idx = torch.argmax(iou_scores, dim=1, keepdim=True)
                idx = idx.reshape(b, 1, 1, 1).expand(b, 1, h, w)
                high_res_logits = torch.gather(high_res_logits,
                                               dim=1,
                                               index=idx)

            num_total_masks = gt_masks.size(0)
            # ===
            with torch.no_grad():
                point_coords = get_uncertain_point_coords_with_randomness(
                    high_res_logits, None, self.num_points,
                    self.oversample_ratio, self.importance_sample_ratio)
                mask_point_targets = point_sample(
                    gt_masks.unsqueeze(1).float(), point_coords).squeeze(1)

            mask_point_preds = point_sample(high_res_logits, point_coords)

            # cls loss
            loss_mask += self.loss_mask(mask_point_preds.reshape(-1),
                                        mask_point_targets.reshape(-1),
                                        avg_factor=num_total_masks *
                                        self.num_points)
            # dice loss
            loss_dice += self.loss_dice(mask_point_preds,
                                        mask_point_targets,
                                        avg_factor=num_total_masks)

            # TODO: loss iou
            # compute predicted mask iou with gt
            # iou_scores
            batch_iou = calc_iou(high_res_logits.squeeze(1), gt_masks)
            loss_iou += self.loss_iou(iou_scores,
                                      batch_iou,
                                      avg_factor=num_total_masks)

            # ===

        loss_dict['loss_mask'] = loss_mask / batch_size
        loss_dict['loss_dice'] = loss_dice / batch_size
        loss_dict['loss_iou'] = loss_iou / batch_size

        return loss_dict

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
                 inputs: dict[torch.Tensor],
                 data_samples: OptSampleList = None,
                 multimask_output: bool = False):
        image_embeddings = self.image_encoder(inputs['image'])[0]

        num_prompts = inputs['boxes'].size(0) // image_embeddings.size(0)
        # Expand per-image data in batch direction to be per-mask
        image_embeddings = torch.repeat_interleave(image_embeddings,
                                                   num_prompts,
                                                   dim=0)

        if random.random() > 0.5:
            inputs['point_coords'] = None
        else:
            inputs['boxes'] = None

        if inputs['point_coords'] is not None:
            points = (inputs['point_coords'], inputs['point_labels'])
        else:
            points = None

        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=points,
            boxes=inputs.get('boxes', None),
            masks=inputs.get('mask_inputs', None))

        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_positional_embeddings=self.prompt_encoder.get_dense_pe(),
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
