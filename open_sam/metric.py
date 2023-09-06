import torch

from mmengine.evaluator import BaseMetric
from mmseg.registry import METRICS


@METRICS.register_module()
class IoU(BaseMetric):

    def process(self, data_batch, data_samples):
        '''
        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples: (Sequence[dict]): A batch of outputs from the model.
        '''
        output, gt_masks = data_samples[0], data_samples[1]['gt_masks']

        intersect = 0.
        union = 0.
        for pred, gt_mask in zip(output, gt_masks):
            pred_masks = pred['masks']

            intersect += (pred_masks == gt_mask).sum()
            union += torch.logical_or(pred_masks, gt_mask).sum()
        iou = (intersect / union).cpu()
        self.results.append(dict(batch_size=len(output),
                                 iou=iou * len(output)))

    def compute_metrics(self, results):
        total_iou = sum(result['iou'] for result in self.results)
        num_samples = sum(result['batch_size'] for result in self.results)
        return dict(iou=total_iou / num_samples)
