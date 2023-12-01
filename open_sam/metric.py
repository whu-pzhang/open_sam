import torch
import numpy as np

from mmengine.evaluator import BaseMetric
from mmseg.evaluation import IoUMetric as _IoUMetric
from mmengine.structures import InstanceData

from .registry import METRICS


@METRICS.register_module()
class IoU(_IoUMetric):

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

        masks_dict = [
            dict(mask=m, label=labels[idx], area=torch.sum(m))
            for idx, m in enumerate(masks)
        ]
        sorted_masks = sorted(masks_dict,
                              key=(lambda x: x['area']),
                              reverse=True)

        seg_map = torch.zeros(size=masks[0].shape,
                              dtype=torch.int,
                              device=masks.device)
        # for idx, mask in enumerate(masks):
        #     seg_map[mask.bool()] = labels[idx].to(torch.int)
        for ann in sorted_masks:
            mask = ann['mask'].bool()
            seg_map[mask] = ann['label'].to(torch.int)

        return seg_map
