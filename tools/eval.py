from typing import Dict, Optional, Sequence

import cv2
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt

from pathlib import Path
import argparse
from tqdm import tqdm

from mmengine.utils import ProgressBar
from mmseg.evaluation import IoUMetric as _IoUMetric


def get_arguments():

    parser = argparse.ArgumentParser()

    parser.add_argument('--pred_path', type=str)
    parser.add_argument('--gt_path', type=str)
    parser.add_argument('--seg_map_suffix', type=str, default='.png')
    parser.add_argument('--num-classes', type=int)
    parser.add_argument('--reduce-zero-label', action='store_true')
    parser.add_argument('--scale', type=int, default=1)

    args = parser.parse_args()
    return args


class IoUMetric(_IoUMetric):

    def __init__(self, class_names, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = len(class_names)
        self.class_names = class_names
        self.dataset_meta = dict(classes=class_names)

    def process(self, gt, pred):
        self.results.append(
            self.intersect_and_union(torch.from_numpy(pred),
                                     torch.from_numpy(gt),
                                     num_classes=self.num_classes,
                                     ignore_index=self.ignore_index))

    def compute_metrics(self) -> Dict[str, float]:
        return super().compute_metrics(self.results)


def main():

    args = get_arguments()
    print("Args:", args, "\n"),

    gt_dir = Path(args.gt_path)
    pred_dir = Path(args.pred_path)

    suffix = ('*.jpg', '*.png', '*.tiff', '*.tif')

    # gt_images = [f for s in suffix for f in gt_dir.rglob(s)]
    pred_images = [f for s in suffix for f in pred_dir.rglob(s)]

    metric = IoUMetric(
        class_names=[
            # 'impervious_surface', 'building', 'low_vegetation', 'tree', 'car',
            # 'clutter'
            'background',
            'building',
            'road',
            'water',
            'barren',
            'forest',
            'agricultural'
        ],
        iou_metrics=['mIoU', 'mFscore'])
    # pbar = tqdm(total=len(gt_images))
    pbar = ProgressBar(task_num=len(pred_images))
    for pred_img in pred_images:
        gt_path = gt_dir / (pred_img.stem + args.seg_map_suffix)
        gt = np.array(Image.open(gt_path))
        gt = (gt // args.scale).astype(np.uint8)

        if args.reduce_zero_label:
            gt[gt == 0] = 255
            gt = gt - 1
            gt[gt == 254] = 255

        pred = np.array(Image.open(pred_img))

        metric.process(gt, pred)

        pbar.update()

    ret_metrics = metric.compute_metrics()
    print(ret_metrics)


if __name__ == '__main__':
    main()
