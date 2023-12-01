from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Union, Sequence
import random

import cv2
import numpy as np
import torch
from torch.nn import functional as F
from torchvision.transforms.functional import resize  # type: ignore
from torchvision.transforms.functional import to_pil_image
from mmcv.transforms import to_tensor, BaseTransform
import mmengine
from mmengine.structures import InstanceData

from mmdet.structures.bbox import HorizontalBoxes, BaseBoxes
from mmdet.structures.mask import BitmapMasks

from open_sam.registry import TRANSFORMS

from .sam_data_sample import SamDataSample


@TRANSFORMS.register_module()
class GenerateSAMPrompt(BaseTransform):
    prompt_types = ('point', 'boxes', 'masks')

    def __init__(self,
                 prompt_type=['point', 'boxes'],
                 max_instances_per_classes=15,
                 points_per_instance=2,
                 noise_cfg=dict(bbox_std_ratio=0.1, bbox_max_offset=20),
                 ignore_values=[255]):
        valid_prompts = ['point', 'boxes']

        if isinstance(prompt_type, str):
            assert prompt_type in valid_prompts
            prompt_type = [prompt_type]
        elif isinstance(prompt_type, list):
            assert mmengine.is_list_of(prompt_type, str)
            assert set(prompt_type).issubset(set(valid_prompts))
        else:
            raise ValueError(f'prompt_type must be either str or list of str, \
                               but got `{type(prompt_type)}`.')
        self.prompt_type = prompt_type

        self.max_instances_per_classes = max_instances_per_classes
        self.points_per_instance = points_per_instance
        self.ignore_values = ignore_values
        self.noise_cfg = noise_cfg

    @property
    def add_noise(self):
        return self.noise_cfg is not None

    @property
    def with_boxes(self):
        return 'boxes' in self.prompt_type

    @property
    def with_point(self):
        return ('point' in self.prompt_type) and self.has_masks

    @property
    def has_masks(self):
        return self._has_masks or self._has_seg_map

    def transform(self, results):
        gt_seg_map = results.get('gt_seg_map', None)
        gt_masks = results.get('gt_masks', None)
        gt_bboxes = results.get('gt_bboxes', None)

        self._has_masks = gt_masks is not None
        self._has_seg_map = gt_seg_map is not None

        assert (gt_bboxes is not None) or (gt_seg_map is not None)

        if (gt_seg_map is not None) and (gt_bboxes is None):
            results = self.segmap2instance(results)

        prompts = self.generate_prompt(
            results,
            max_instances=self.max_instances_per_classes,
            points_per_instance=self.points_per_instance,
            noise_cfg=self.noise_cfg)

        # 0:point, 1:box, 2:mask
        prompts['prompt_type'] = random.choice(self.prompt_type)
        results.update(prompts)
        return results

    @staticmethod
    def segmap2instance(results, ignore_values=[0, 255]):
        img_h, img_w = results['img_shape']
        gt_masks = []
        gt_bboxes = []
        gt_bboxes_labels = []

        seg_map = results['gt_seg_map']
        classes = np.unique(seg_map, return_inverse=False, return_counts=False)
        for val in classes:
            # remove ignored region
            if val in ignore_values:
                continue
            cur_class_mask = (seg_map == val).astype(np.uint8)
            cur_class_mask = cv2.morphologyEx(cur_class_mask,
                                              cv2.MORPH_OPEN,
                                              kernel=np.ones((3, 3), np.uint8))
            num_objects, labels, stats, centroids = cv2.connectedComponentsWithStats(
                cur_class_mask, connectivity=4)

            for i in range(1, num_objects):
                cur_object = (labels == i)
                box_x = stats[i, cv2.CC_STAT_LEFT]
                box_y = stats[i, cv2.CC_STAT_TOP]
                box_w = stats[i, cv2.CC_STAT_WIDTH]
                box_h = stats[i, cv2.CC_STAT_HEIGHT]

                # gt_bboxes
                gt_bboxes.append(
                    [box_x, box_y, box_x + box_w - 1, box_y + box_h - 1])
                gt_bboxes_labels.append(val)
                # gt_mask_instance
                gt_masks.append(cur_object.astype(np.uint8))

        gt_masks = BitmapMasks(gt_masks, img_h, img_w)
        gt_bboxes = HorizontalBoxes(gt_bboxes, in_mode='xyxy')
        gt_bboxes_labels = np.array(gt_bboxes_labels)

        results['gt_masks'] = gt_masks
        results['gt_bboxes'] = gt_bboxes
        results['gt_bboxes_labels'] = gt_bboxes_labels

        # add img_id for seg dataset
        results['img_id'] = results['sample_idx']

        return results

    def generate_prompt(self,
                        results,
                        max_instances: int = 15,
                        points_per_instance: int = 2,
                        noise_cfg: dict = None):
        '''
        First, with equal probability either a foreground point or bounding box
        is selected randomly for the target mask. Points are sampled uniformly
        from the ground truth mask.
        Boxes are taken as the ground truth mask’s bounding box,
        with random noise added in each coordinate with standard deviation
        equal to 10% of the box sidelength, to a maximum of 20 pixels.

        results is a dict:
            gt_bboxes: BaseBoxes(N,4) xyxy format int type
            gt_bboxes_labels: np.ndarray(N, )
            gt_masks: BitmapMasks (H,W) uint8 type
            gt_seg_map(optional): np.darray(H, W) float type

        Added keys:
            boxes:
            masks:
            point_coords
        '''
        img_h, img_w = results['img_shape']
        gt_masks_old = results.get('gt_masks', None)
        gt_bboxes_old = results['gt_bboxes'].numpy()
        gt_bboxes_labels_old = results['gt_bboxes_labels']
        if self.has_masks:
            gt_masks_old = results['gt_masks'].to_ndarray()
            gt_masks = []
            if self.with_point:
                point_coords = []

        if self.add_noise:
            bbox_std_ratio = noise_cfg['bbox_std_ratio']
            bbox_max_offset = noise_cfg['bbox_max_offset']

        #1. random select ground truth masks as prompt
        num_gts = len(gt_bboxes_old)
        max_num_objects = min(max_instances, num_gts)
        random_selected_idx = np.random.choice(num_gts,
                                               size=max_num_objects,
                                               replace=False)

        if self.with_boxes:
            gt_bboxes = []
            gt_bboxes_labels = []
            boxes = []

        for idx in random_selected_idx:
            if self.with_boxes:
                # add noise to bbox
                cur_bbox = gt_bboxes_old[idx]
                cur_label = gt_bboxes_labels_old[idx]
                gt_bboxes.append(cur_bbox)
                gt_bboxes_labels.append(cur_label)

                if self.add_noise:
                    w, h = cur_bbox[2] - cur_bbox[0], cur_bbox[3] - cur_bbox[1]
                    x_noise = np.random.normal(0,
                                               scale=w * bbox_std_ratio,
                                               size=2)
                    y_noise = np.random.normal(0,
                                               scale=h * bbox_std_ratio,
                                               size=2)
                    bbox_noise = np.array(
                        [x_noise[0], y_noise[0], x_noise[1], y_noise[1]])
                    bbox_noise = np.clip(bbox_noise, -bbox_max_offset,
                                         bbox_max_offset)
                    box = cur_bbox + bbox_noise
                    boxes.append(box)
                else:
                    boxes.append(cur_bbox)

            cur_mask = gt_masks_old[idx]
            gt_masks.append(cur_mask)
            if self.with_point:
                # sample points
                y, x = torch.meshgrid(torch.arange(img_h), torch.arange(img_w))
                x_idx = torch.masked_select(
                    x, torch.as_tensor(cur_mask, dtype=torch.bool))
                y_idx = torch.masked_select(
                    y, torch.as_tensor(cur_mask, dtype=torch.bool))
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

        out = dict()
        gt_masks = np.stack(gt_masks)
        out.update(gt_masks=gt_masks)

        if self.with_boxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_bboxes_labels = np.array(gt_bboxes_labels, dtype=np.int64)
            boxes = np.array(boxes, dtype=np.float32)
            # clip
            boxes[..., 0::2] = boxes[..., 0::2].clip(0, img_w)
            boxes[..., 1::3] = boxes[..., 1::3].clip(0, img_h)
            out.update(gt_bboxes=gt_bboxes,
                       gt_bboxes_labels=gt_bboxes_labels,
                       boxes=boxes)

        if self.with_point:
            point_coords = np.stack(point_coords)
            out.update(point_coords=point_coords)

        return out


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
class PackSamInputs(BaseTransform):

    mapping_table = {
        'gt_bboxes': 'bboxes',
        'gt_bboxes_labels': 'labels',
        'gt_masks': 'masks'
    }

    def __init__(self,
                 meta_keys: Optional[dict] = None,
                 default_meta_keys=('img_id', 'img_path', 'ori_shape',
                                    'img_shape', 'scale_factor', 'flip',
                                    'flip_direction', 'reduce_zero_label')):
        self.meta_keys = default_meta_keys
        if meta_keys is not None:
            if isinstance(meta_keys, str):
                meta_keys = (meta_keys, )
            else:
                assert isinstance(meta_keys, tuple), \
                    'meta_keys must be str or tuple'
            self.meta_keys += meta_keys

    def transform(self, results):
        packed_results = dict()
        if 'img' in results:
            img = results['img']

            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            # To improve the computational speed by by 3-5 times, apply:
            # If image is not contiguous, use
            # `numpy.transpose()` followed by `numpy.ascontiguousarray()`
            # If image is already contiguous, use
            # `torch.permute()` followed by `torch.contiguous()`
            # Refer to https://github.com/open-mmlab/mmdetection/pull/9533
            # for more details
            if not img.flags.c_contiguous:
                img = to_tensor(np.ascontiguousarray(img.transpose(2, 0, 1)))
            else:
                img = to_tensor(img).permute(2, 0, 1).contiguous()

            # inputs['img'] = img
            packed_results['inputs'] = img

        # 2. Pack InstanceData
        data_sample = SamDataSample()
        gt_instance = InstanceData()
        prompt_instance = InstanceData()

        assert 'img_id' in results, "'img_id' must contained in the results "
        'for counting the number of images'

        for key in self.mapping_table.keys():
            if key not in results:
                continue
            if key == 'gt_masks' or isinstance(results[key], BaseBoxes):
                gt_instance[self.mapping_table[key]] = to_tensor(results[key])
            else:
                gt_instance[self.mapping_table[key]] = to_tensor(results[key])

        if results.get('point_coords', None) is not None:
            point_coords = to_tensor(results['point_coords'])
            point_labels = torch.ones(point_coords.shape[:2],
                                      dtype=torch.uint8)
            prompt_instance['point_coords'] = point_coords
            prompt_instance['point_labels'] = point_labels

        boxes = to_tensor(results['boxes'].astype(np.float32))
        prompt_instance['boxes'] = boxes

        # ---
        # print(len(boxes), len(point_coords), len(point_labels), len(masks))
        data_sample.gt_instances = gt_instance
        data_sample.prompt_instances = prompt_instance
        # ---
        # 3. Pack img_meta
        img_meta = dict(prompt_type=results['prompt_type'],
                        original_size=results['img_shape'])
        for key in self.meta_keys:
            if key in results:
                img_meta[key] = results[key]
        data_sample.set_metainfo(img_meta)

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
