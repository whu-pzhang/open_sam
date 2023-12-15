from typing import Sequence
import torch
import numpy as np

from mmengine.evaluator import BaseMetric
from mmseg.evaluation import IoUMetric as _IoUMetric
from mmengine.structures import InstanceData
from mmdet.structures.mask import BitmapMasks

from .registry import METRICS


@METRICS.register_module()
class ClassAwareIoU(_IoUMetric):

    def process(self, data_batch, data_samples):
        '''
        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples: (Sequence[dict]): A batch of outputs from the model.
        '''
        num_classes = len(self.dataset_meta['classes'])
        for data_sample in data_samples:
            pred = data_sample['pred_instances']
            gt = data_sample['gt_instances']

            # specify label for each mask
            pred_seg_map = self.instance2segmap(pred)
            gt_seg_map = self.instance2segmap(gt)

            self.results.append(
                self.intersect_and_union(pred_seg_map, gt_seg_map, num_classes,
                                         self.ignore_index))

    def instance2segmap(self, instance: InstanceData):
        masks = instance['masks']
        labels = instance['labels']

        seg_map = torch.zeros(size=masks.shape[1:],
                              dtype=torch.int,
                              device=masks.device)
        if len(labels) == 0 or labels[0] is None:
            return seg_map

        masks_dict = [
            dict(mask=m, label=labels[idx], area=torch.sum(m))
            for idx, m in enumerate(masks)
        ]
        sorted_masks = sorted(masks_dict,
                              key=(lambda x: x['area']),
                              reverse=True)

        # for idx, mask in enumerate(masks):
        #     seg_map[mask.bool()] = labels[idx].to(torch.int)
        for ann in sorted_masks:
            mask = ann['mask'].bool()
            seg_map[mask] = ann['label'].to(torch.int)

        return seg_map


@METRICS.register_module()
class ClassAgnosticIoU(_IoUMetric):

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        num_classes = 2
        for data_sample in data_samples:
            pred = data_sample['pred_instances']
            gt = data_sample['gt_instances']

            # specify label for each mask
            pred_seg_map = self.instance2segmap(pred)
            gt_seg_map = self.instance2segmap(gt)

            # import matplotlib.pyplot as plt
            # from pathlib import Path
            # f, ax = plt.subplots(1, 2)
            # ax[0].imshow(gt_seg_map.cpu().numpy())
            # ax[1].imshow(pred_seg_map.cpu().numpy())
            # plt.title(Path(data_sample['img_path']).name)
            # plt.show()

            self.results.append(
                self.intersect_and_union(pred_seg_map, gt_seg_map, num_classes,
                                         self.ignore_index))

    def instance2segmap(self, instance: InstanceData):
        masks = instance['masks']
        if isinstance(masks, BitmapMasks):
            masks = masks.to_tensor(dtype=torch.uint8, device='cpu')

        seg_map = torch.zeros(size=masks.shape[1:], dtype=torch.uint8)
        masks_dict = [
            dict(mask=m, area=m.sum()) for idx, m in enumerate(masks)
        ]
        sorted_masks = sorted(masks_dict,
                              key=(lambda x: x['area']),
                              reverse=True)
        for ann in sorted_masks:
            mask = ann['mask'].bool()
            seg_map[mask] = 1

        return seg_map
