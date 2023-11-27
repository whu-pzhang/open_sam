from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Union
import random

import cv2
import numpy as np
import torch
from torch.nn import functional as F
from torchvision.transforms.functional import resize  # type: ignore
from torchvision.transforms.functional import to_pil_image
from mmcv.transforms import to_tensor, BaseTransform
from mmengine.structures import PixelData, InstanceData
from mmdet.structures.mask import BitmapMasks

from open_sam.registry import TRANSFORMS

from .sam_data_sample import SamDataSample


@TRANSFORMS.register_module()
class GenerateSAMPrompt(BaseTransform):
    prompt_types = ('point', 'boxes', 'masks')

    def __init__(self,
                 max_instances_per_classes=15,
                 points_per_instance=2,
                 ignore_values=[255]):

        self.max_instances_per_classes = max_instances_per_classes
        self.points_per_instance = points_per_instance
        self.ignore_values = ignore_values

    def transform(self, results):
        gt_seg_map = results.get('gt_seg_map', None)
        gt_masks = results.get('gt_masks', None)

        if gt_masks is not None:
            propmts = self.generate_prompt_from_coco(
                results, self.max_instances_per_classes,
                self.points_per_instance)
        elif gt_seg_map is not None:
            propmts = self.generate_prompt_from_seg_map(
                gt_seg_map, self.max_instances_per_classes,
                self.points_per_instance, self.ignore_values)

        # 0:point, 1:box, 2:mask
        propmts['prompt_type'] = random.choice(range(2))
        results.update(propmts)
        return results

    @staticmethod
    def generate_prompt_from_seg_map(seg_map,
                                     max_instances=15,
                                     points_per_instance=2,
                                     ignore_values=[0, 255]):
        img_h, img_w = seg_map.shape[:2]
        gt_masks = []
        boxes = []
        point_coords = []
        class_vals, counts = np.unique(seg_map, return_counts=True)
        for val in class_vals:
            if val in ignore_values:
                continue
            cur_class_mask = (seg_map == val).astype(np.uint8)
            cur_class_mask = cv2.morphologyEx(cur_class_mask,
                                              cv2.MORPH_OPEN,
                                              kernel=np.ones((3, 3), np.uint8))
            num_objects, labels, stats, centroids = cv2.connectedComponentsWithStats(
                cur_class_mask, connectivity=4)

            try:
                object_idxs = random.sample(range(1, num_objects),
                                            k=max_instances)
            except:
                object_idxs = range(1, num_objects)

            for i in object_idxs:
                cur_object = (labels == i)
                box_x = stats[i, cv2.CC_STAT_LEFT]
                box_y = stats[i, cv2.CC_STAT_TOP]
                box_w = stats[i, cv2.CC_STAT_WIDTH]
                box_h = stats[i, cv2.CC_STAT_HEIGHT]

                # sample points
                y, x = torch.meshgrid(torch.arange(0, img_h),
                                      torch.arange(0, img_w))
                x_idx = torch.masked_select(x, torch.as_tensor(cur_object))
                y_idx = torch.masked_select(y, torch.as_tensor(cur_object))
                if len(x_idx) < points_per_instance:
                    continue
                selected_idx = torch.randperm(
                    x_idx.shape[0])[:points_per_instance]
                # print(len(x_idx), len(selected_idx))
                samples_x = x_idx[selected_idx].numpy()
                samples_y = y_idx[selected_idx].numpy()
                samples_xy = np.concatenate(
                    [samples_x[:, None], samples_y[:, None]], axis=1)
                point_coords.append(samples_xy)

                # boxes
                boxes.append(
                    [box_x, box_y, box_x + box_w - 1, box_y + box_h - 1])

                # gt_mask_instance
                gt_masks.append(cur_object.astype(np.uint8))

        if len(gt_masks) == 0:
            gt_masks, point_coords, boxes = None, None, None
        else:
            gt_masks = np.stack(gt_masks)
            point_coords = np.stack(point_coords)
            boxes = np.stack(boxes)

        out = dict(gt_masks=gt_masks, point_coords=point_coords, boxes=boxes)
        return out

    def generate_prompt_from_coco(self,
                                  results: dict,
                                  max_instances: int = 15,
                                  points_per_instance: int = 2):
        img_h, img_w = results['img_shape']
        gt_bboxes = results['gt_bboxes'].numpy()  # BaseBoxes
        bitmap_masks = results['gt_masks'].to_ndarray()  # BitmapMasks

        gt_masks = []
        boxes = []
        point_coords = []

        object_idxs = list(range(len(bitmap_masks)))
        random.shuffle(object_idxs)

        num_objects = min(max_instances, len(bitmap_masks))
        # object_idxs = object_idxs[:num_objects]
        object_idxs = np.random.choice(object_idxs,
                                       size=num_objects,
                                       replace=False)

        keep_idx = []

        for i in object_idxs:
            mask = bitmap_masks[i]
            # sample points
            y, x = torch.meshgrid(torch.arange(img_h), torch.arange(img_w))
            x_idx = torch.masked_select(
                x, torch.as_tensor(mask, dtype=torch.bool))
            y_idx = torch.masked_select(
                y, torch.as_tensor(mask, dtype=torch.bool))
            if len(x_idx) < points_per_instance:
                continue
            selected_idx = torch.randperm(x_idx.shape[0])[:points_per_instance]
            # print(len(x_idx), len(selected_idx))
            samples_x = x_idx[selected_idx].numpy()
            samples_y = y_idx[selected_idx].numpy()
            samples_xy = np.concatenate(
                [samples_x[:, None], samples_y[:, None]], axis=1)
            point_coords.append(samples_xy)

            gt_masks.append(mask)
            boxes.append(gt_bboxes[i])

            keep_idx.append(i)

        gt_masks = np.stack(gt_masks)
        point_coords = np.stack(point_coords)
        boxes = np.stack(boxes)
        out = dict(gt_masks=gt_masks, point_coords=point_coords, boxes=boxes)
        return out

    def add_noise(self, prompt):
        '''Boxes are taken as the ground truth mask’s bounding box,
        with random noise added in each coordinate with standard deviation
        equal to 10% of the box sidelength, to a maximum of 20 pixels.

        prompt is a dict:
            gt_masks:
            gt_bboxes: xyxy format
            point_coords:

        add key:
            boxes:
            masks:
        '''


@TRANSFORMS.register_module()
class ResizeLongestEdge(BaseTransform):

    def __init__(self, scale: Union[int, Tuple[int, int]]):
        self.scale = scale
        self.resize = TRANSFORMS.build(
            dict(type='mmdet.Resize', scale=0, keep_ratio=True))

    def _get_output_shape(self, img, long_edge_length) -> Tuple[int, int]:
        """Compute the target image shape with the given `short_edge_length`.

        Args:
            img (np.ndarray): The input image.
            short_edge_length (Union[int, Tuple[int, int]]): The target short
                edge length. If it's tuple, will select the min value as the
                short edge length.
        """
        h, w = img.shape[:2]
        if isinstance(long_edge_length, int):
            size = long_edge_length * 1.0
        elif isinstance(long_edge_length, tuple):
            size = max(long_edge_length) * 1.0
        scale = size / max(h, w)
        if h < w:
            new_h, new_w = size, scale * w
        else:
            new_h, new_w = scale * h, size

        new_h = int(new_h + 0.5)
        new_w = int(new_w + 0.5)
        return (new_w, new_h)

    def transform(self, results):
        self.resize.scale = self._get_output_shape(results['img'], self.scale)
        return self.resize(results)


@TRANSFORMS.register_module()
class ResizeLongestSide:
    """Resizes images to longest side 'target_length', as well as provides
    methods for resizing coordinates and boxes.

    Provides methods for transforming both numpy array and batched torch
    tensors.
    """

    def __init__(self, target_length: int) -> None:
        self.target_length = target_length

    def __call__(self, results):
        ori_shape = results['img_shape']
        results['img'] = self.apply_image(results['img'])
        results['gt_masks'] = self.apply_mask(results['gt_masks'])
        results['boxes'] = self.apply_boxes(results['boxes'], ori_shape)
        results['point_coords'] = self.apply_coords(results['point_coords'],
                                                    ori_shape)
        results['img_shape'] = results['img'].shape[:2]

        return results

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        """Expects a numpy array with shape HxWxC in uint8 format."""
        target_size = self.get_preprocess_shape(image.shape[0], image.shape[1],
                                                self.target_length)
        return np.array(resize(to_pil_image(image), target_size))

    def apply_mask(self, masks: np.ndarray):
        ret_masks = [self.apply_image(m) for m in masks]
        return np.stack(ret_masks)

    def apply_coords(self, coords: np.ndarray,
                     original_size: Tuple[int, ...]) -> np.ndarray:
        """Expects a numpy array of length 2 in the final dimension.

        Requires the original image size in (H, W) format.
        """
        old_h, old_w = original_size
        new_h, new_w = self.get_preprocess_shape(original_size[0],
                                                 original_size[1],
                                                 self.target_length)
        coords = deepcopy(coords).astype(float)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords

    def apply_boxes(self, boxes: np.ndarray,
                    original_size: Tuple[int, ...]) -> np.ndarray:
        """Expects a numpy array shape Bx4.

        Requires the original image size in (H, W) format.
        """
        boxes = self.apply_coords(boxes.reshape(-1, 2, 2), original_size)
        return boxes.reshape(-1, 4)

    def apply_image_torch(self, image: torch.Tensor) -> torch.Tensor:
        """Expects batched images with shape BxCxHxW and float format.

        This transformation may not exactly match apply_image. apply_image is
        the transformation expected by the model.
        """
        # Expects an image in BCHW format. May not exactly match apply_image.
        target_size = self.get_preprocess_shape(image.shape[0], image.shape[1],
                                                self.target_length)
        return F.interpolate(image,
                             target_size,
                             mode='bilinear',
                             align_corners=False,
                             antialias=True)

    def apply_coords_torch(self, coords: torch.Tensor,
                           original_size: Tuple[int, ...]) -> torch.Tensor:
        """Expects a torch tensor with length 2 in the last dimension.

        Requires the original image size in (H, W) format.
        """
        old_h, old_w = original_size
        new_h, new_w = self.get_preprocess_shape(original_size[0],
                                                 original_size[1],
                                                 self.target_length)
        coords = deepcopy(coords).to(torch.float)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords

    def apply_boxes_torch(self, boxes: torch.Tensor,
                          original_size: Tuple[int, ...]) -> torch.Tensor:
        """Expects a torch tensor with shape Bx4.

        Requires the original image size in (H, W) format.
        """
        boxes = self.apply_coords_torch(boxes.reshape(-1, 2, 2), original_size)
        return boxes.reshape(-1, 4)

    @staticmethod
    def get_preprocess_shape(oldh: int, oldw: int,
                             long_side_length: int) -> Tuple[int, int]:
        """Compute the output size given input size and target long side
        length."""
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)


@TRANSFORMS.register_module()
class Pad:

    def __init__(self):
        pass

    def transform(self, results):
        pass

    def __call__(self, results):
        return self.transform(results)


@TRANSFORMS.register_module()
class PackSamInputs(BaseTransform):

    def __init__(self,
                 meta_keys=('img_path', 'seg_map_path', 'ori_shape',
                            'img_shape', 'pad_shape', 'scale_factor', 'flip',
                            'flip_direction', 'reduce_zero_label')):
        self.meta_keys = meta_keys

    def transform(self, results):
        packed_results = dict()
        if 'img' in results:
            img = results['img']

            inputs = dict()
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            if not img.flags.c_contiguous:
                img = to_tensor(np.ascontiguousarray(img.transpose(2, 0, 1)))
            else:
                img = img.transpose(2, 0, 1)
                img = to_tensor(img).contiguous()

            inputs['img'] = img

        data_sample = SamDataSample()
        instance_data = InstanceData()
        # data = to_tensor(results['gt_masks'].astype(np.int64))
        # gt_masks_data = dict(data=data)
        # data_sample.gt_masks = PixelData(**gt_masks_data)

        gt_masks = results['gt_masks']
        if gt_masks is not None:
            point_coords = to_tensor(results['point_coords'])
            point_labels = torch.ones(point_coords.shape[:2],
                                      dtype=torch.uint8)
            boxes = to_tensor(results['boxes'])
            masks = to_tensor(results['gt_masks'].astype(np.int64))

            instance_data['boxes'] = boxes
            instance_data['point_coords'] = point_coords
            instance_data['point_labels'] = point_labels
            instance_data['masks'] = masks

        # ---
        # print(len(boxes), len(point_coords), len(point_labels), len(masks))
        data_sample.gt_instances = instance_data
        # ---

        img_meta = dict(prompt_type=results['prompt_type'],
                        original_size=results['img_shape'])

        data_sample.set_metainfo(img_meta)

        packed_results['inputs'] = inputs
        packed_results['data_samples'] = data_sample

        return packed_results


class FilterAnnotations(BaseTransform):
    '''

        - gt_masks
        - point_coords
        - boxes
    '''

    def __init__(self,
                 min_gt_bbox_wh: Tuple[int, int] = (1, 1),
                 min_gt_mask_area: int = 1,
                 by_box: bool = True,
                 by_mask: bool = False,
                 keep_empty: bool = True) -> None:
        super().__init__()
        assert by_box or by_mask
        self.min_gt_bbox_wh = min_gt_bbox_wh
        self.min_gt_mask_area = min_gt_mask_area
        self.by_box = by_box
        self.by_mask = by_mask
        self.keep_empty = keep_empty

    def transform(self, results: Dict) -> Union[dict, None]:

        gt_masks = results['gt_masks']
