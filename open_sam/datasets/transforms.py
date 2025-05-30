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
from mmcv.transforms import LoadAnnotations as MMCV_LoadAnnotations
import mmengine
from mmengine.structures import InstanceData

from mmdet.structures.bbox import HorizontalBoxes, BaseBoxes, get_box_type
from mmdet.structures.mask import BitmapMasks
import pycocotools.mask as maskUtils

from open_sam.registry import TRANSFORMS
from .sam_data_sample import SamDataSample


def point_sampling(mask, num_points=2):

    fg_coords = np.argwhere(mask > 0)[:, ::-1]
    bg_coords = np.argwhere(mask == 0)[:, ::-1]
    fg_size = len(fg_coords)
    bg_size = len(bg_coords)

    num_bg = num_points // 2
    num_fg = num_points - num_bg
    fg_indices = np.random.choice(fg_size, size=num_fg, replace=True)
    bg_indices = np.random.choice(bg_size, size=num_bg, replace=True)
    fg_coords = fg_coords[fg_indices]
    bg_coords = bg_coords[bg_indices]
    coords = np.concatenate([fg_coords, bg_coords], axis=0)
    labels = np.concatenate([np.ones(num_fg), np.zeros(num_bg)]).astype(int)
    indices = np.random.permutation(num_points)
    return coords[indices], labels[indices]


def bbox_perturbing(bbox, std_ratio=0.1, max_offset=20):
    '''
        bbox: (xyxy format)
    '''
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x_noise = np.random.normal(0, scale=w * std_ratio, size=2)
    y_noise = np.random.normal(0, scale=h * std_ratio, size=2)
    bbox_noise = np.array([x_noise[0], y_noise[0], x_noise[1], y_noise[1]])
    bbox_noise = np.clip(bbox_noise, -max_offset, max_offset)
    bbox = bbox + bbox_noise
    return bbox


@TRANSFORMS.register_module()
class LoadAnnotations(MMCV_LoadAnnotations):

    def __init__(self,
                 with_mask=True,
                 box_type: str = 'hbox',
                 ignore_index=255,
                 **kwargs):
        super().__init__(**kwargs)

        self.with_mask = with_mask
        self.box_type = box_type
        self.ignore_index = ignore_index

    def _load_bboxes(self, results: dict) -> None:
        """Private function to load bounding box annotations.

        Args:
            results (dict): Result dict from :obj:``mmengine.BaseDataset``.
        Returns:
            dict: The dict contains loaded bounding box annotations.
        """
        gt_bboxes = []
        gt_ignore_flags = []
        for instance in results.get('instances', []):
            gt_bboxes.append(instance['bbox'])
            gt_ignore_flags.append(instance['ignore_flag'])
        if self.box_type is None:
            results['gt_bboxes'] = np.array(gt_bboxes,
                                            dtype=np.float32).reshape((-1, 4))
        else:
            _, box_type_cls = get_box_type(self.box_type)
            results['gt_bboxes'] = box_type_cls(gt_bboxes, dtype=torch.float32)
        results['gt_ignore_flags'] = np.array(gt_ignore_flags, dtype=bool)

    def _poly2mask(self, mask_ann: Union[list, dict], img_h: int,
                   img_w: int) -> np.ndarray:
        """Private function to convert masks represented with polygon to
        bitmaps.

        Args:
            mask_ann (list | dict): Polygon mask annotation input.
            img_h (int): The height of output mask.
            img_w (int): The width of output mask.

        Returns:
            np.ndarray: The decode bitmap mask of shape (img_h, img_w).
        """

        if isinstance(mask_ann, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(mask_ann, img_h, img_w)
            rle = maskUtils.merge(rles)
        elif isinstance(mask_ann['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(mask_ann, img_h, img_w)
        else:
            # rle
            rle = mask_ann
        mask = maskUtils.decode(rle)
        return mask

    def _process_masks(self, results: dict) -> list:
        """Process gt_masks and filter invalid polygons.

        Args:
            results (dict): Result dict from :obj:``mmengine.BaseDataset``.

        Returns:
            list: Processed gt_masks.
        """
        gt_masks = []
        gt_ignore_flags = []
        for instance in results.get('instances', []):
            gt_mask = instance['mask']
            # If the annotation of segmentation mask is invalid,
            # ignore the whole instance.
            if isinstance(gt_mask, list):
                gt_mask = [
                    np.array(polygon) for polygon in gt_mask
                    if len(polygon) % 2 == 0 and len(polygon) >= 6
                ]
                if len(gt_mask) == 0:
                    # ignore this instance and set gt_mask to a fake mask
                    instance['ignore_flag'] = 1
                    gt_mask = [np.zeros(6)]
            elif isinstance(gt_mask, dict) and \
                    not (gt_mask.get('counts') is not None and
                         gt_mask.get('size') is not None and
                         isinstance(gt_mask['counts'], (list, str))):
                # if gt_mask is a dict, it should include `counts` and `size`,
                # so that `BitmapMasks` can uncompressed RLE
                instance['ignore_flag'] = 1
                gt_mask = [np.zeros(6)]
            gt_masks.append(gt_mask)
            # re-process gt_ignore_flags
            gt_ignore_flags.append(instance['ignore_flag'])
        results['gt_ignore_flags'] = np.array(gt_ignore_flags, dtype=bool)
        return gt_masks

    def _load_masks(self, results: dict) -> None:
        """Private function to load mask annotations.

        Args:
            results (dict): Result dict from :obj:``mmengine.BaseDataset``.
        """
        h, w = results['ori_shape']
        gt_masks = self._process_masks(results)
        gt_masks = BitmapMasks(
            [self._poly2mask(mask, h, w) for mask in gt_masks], h, w)

        results['gt_masks'] = gt_masks

    def transform(self, results: dict) -> dict:
        """Function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:``mmengine.BaseDataset``.

        Returns:
            dict: The dict contains loaded bounding box, label and
            semantic segmentation.
        """

        if self.with_bbox:
            self._load_bboxes(results)
        if self.with_mask:
            self._load_masks(results)
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(with_bbox={self.with_bbox}, '
        repr_str += f'with_mask={self.with_mask}, '
        repr_str += f"imdecode_backend='{self.imdecode_backend}', "
        repr_str += f'backend_args={self.backend_args})'
        return repr_str


@TRANSFORMS.register_module()
class GenerateSAMPrompt(BaseTransform):
    valid_prompts = ['point', 'bbox']

    def __init__(self,
                 prompt_type=['point', 'bbox'],
                 max_instances_per_classes=15,
                 points_per_instance=2,
                 noise_cfg=dict(bbox_std_ratio=0.1, bbox_max_offset=20),
                 ignore_values=[255]):
        if isinstance(prompt_type, str):
            assert prompt_type in self.valid_prompts
            prompt_type = [prompt_type]
        elif isinstance(prompt_type, list):
            assert mmengine.is_list_of(prompt_type, str)
            assert set(prompt_type).issubset(set(self.valid_prompts))
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

    def transform(self, results):
        gt_masks = results.get('gt_masks', None)
        gt_bboxes = results.get('gt_bboxes', None)

        self._has_masks = gt_masks is not None

        assert (gt_bboxes is not None)

        results = self.generate_prompt(
            results,
            max_instances=self.max_instances_per_classes,
            points_per_instance=self.points_per_instance,
            noise_cfg=self.noise_cfg)
        results.update(prompt_type=self.prompt_type)
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

        Added keys:
            boxes:
            masks:
            point_coords
        '''
        gt_bboxes = results['gt_bboxes']  # HorizontalBoxes
        # gt_bboxes_labels_old = results['gt_bboxes_labels']
        gt_masks = results['gt_masks']  # Bitmaps
        if not gt_bboxes.numel:
            return results

        if self.add_noise:
            bbox_std_ratio = noise_cfg['bbox_std_ratio']
            bbox_max_offset = noise_cfg['bbox_max_offset']

        #1. random select ground truth masks as prompt
        num_gts = len(gt_bboxes)
        # replace=True to ensure the number of prompts is max_instances
        random_selected_idx = np.random.choice(num_gts,
                                               size=min(
                                                   max_instances, num_gts),
                                               replace=False)
        keep_idxs = []
        boxes = []
        point_coords = []
        point_labels = []
        for idx in random_selected_idx:
            cur_mask = gt_masks[idx].to_ndarray().squeeze()
            cur_bbox = gt_bboxes[idx].numpy().squeeze()
            if not np.any(cur_mask):  # ensure mask valid
                continue
            if self.add_noise:
                cur_bbox = bbox_perturbing(cur_bbox,
                                           std_ratio=bbox_std_ratio,
                                           max_offset=bbox_max_offset)
            boxes.append(cur_bbox)
            coords, labels = point_sampling(cur_mask,
                                            num_points=points_per_instance)
            point_coords.append(coords)
            point_labels.append(labels)
            keep_idxs.append(idx)
        results.update(gt_bboxes=gt_bboxes[keep_idxs],
                       gt_masks=gt_masks[keep_idxs],
                       gt_ignore_flags=results['gt_ignore_flags'][keep_idxs])

        # prompt boxes
        boxes = np.array(boxes, dtype=np.float32)
        boxes[..., 0::2] = boxes[..., 0::2].clip(0, results['img_shape'][1])
        boxes[..., 1::2] = boxes[..., 1::2].clip(0, results['img_shape'][0])

        # prompt point
        point_coords = np.stack(point_coords, axis=0).astype(np.float32)
        point_labels = np.stack(point_labels, axis=0).astype(np.int64)
        results.update(boxes=boxes,
                       point_coords=point_coords,
                       point_labels=point_labels)

        return results


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
        '''
        Args:
            resutls (dict): Result dict from the data pipeline.

        Returns:
            dict:
            - 'inputs' (torch.Tensor): The forward data of models.
            - 'data_sample' (SamDataSample): The annotation info of the sample.
        '''
        packed_results = dict()
        inputs = dict()
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

            inputs['image'] = img

        if 'point_coords' in results:
            inputs['point_coords'] = to_tensor(results['point_coords'])
            inputs['point_labels'] = to_tensor(results['point_labels'])
        if 'boxes' in results:
            inputs['boxes'] = to_tensor(results['boxes'])

        inputs['prompt_type'] = results['prompt_type']
        packed_results['inputs'] = inputs

        if 'gt_ignore_flags' in results:
            valid_idx = np.where(results['gt_ignore_flags'] == 0)[0]
            ignore_idx = np.where(results['gt_ignore_flags'] == 1)[0]

        # 2. Pack InstanceData
        data_sample = SamDataSample()
        instance_data = InstanceData()
        ignore_instance_data = InstanceData()

        assert 'img_id' in results, "'img_id' must contained in the results "
        'for counting the number of images'

        for key in self.mapping_table.keys():
            if key not in results:
                continue
            if key == 'gt_masks' or isinstance(results[key], BaseBoxes):
                if 'gt_ignore_flags' in results:
                    instance_data[
                        self.mapping_table[key]] = results[key][valid_idx]
                    ignore_instance_data[
                        self.mapping_table[key]] = results[key][ignore_idx]
                else:
                    instance_data[self.mapping_table[key]] = results[key]
            else:
                if 'gt_ignore_flags' in results:
                    instance_data[
                        self.mapping_table[key]] = results[key][valid_idx]
                    ignore_instance_data[
                        self.mapping_table[key]] = results[key][ignore_idx]
                else:
                    instance_data[self.mapping_table[key]] = to_tensor(
                        results[key])

        data_sample.gt_instances = instance_data
        data_sample.ignore_instance = ignore_instance_data

        # ---
        # 3. Pack img_meta
        img_meta = dict()
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
