import copy
import os
import os.path as osp
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Union

import mmengine
import mmengine.fileio as fileio
import numpy as np
from mmseg.datasets import BaseSegDataset
from mmdet.datasets import BaseDetDataset, CocoDataset

from pycocotools.coco import COCO
import cv2

from open_sam.registry import DATASETS


@DATASETS.register_module()
class SegDataset(BaseSegDataset):
    METAINFO = dict(classes=('bg', 'fg'), palette=[[0, 0, 0], [255, 255, 255]])


# class COCODataset(Dataset):

#     def __init__(self, root_dir, annotation_file, transform=None):
#         self.root_dir = root_dir
#         self.transform = transform
#         self.coco = COCO(annotation_file)
#         self.image_ids = list(self.coco.imgs.keys())

#         # Filter out image_ids without any annotations
#         self.image_ids = [
#             image_id for image_id in self.image_ids
#             if len(self.coco.getAnnIds(imgIds=image_id)) > 0
#         ]

#     def __len__(self):
#         return len(self.image_ids)

#     def __getitem__(self, idx):
#         image_id = self.image_ids[idx]
#         image_info = self.coco.loadImgs(image_id)[0]
#         image_path = osp.join(self.root_dir, image_info['file_name'])
#         image = cv2.imread(image_path)
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#         ann_ids = self.coco.getAnnIds(imgIds=image_id)
#         anns = self.coco.loadAnns(ann_ids)
#         boxes = []
#         masks = []

#         for ann in anns:
#             x, y, w, h = ann['bbox']
#             boxes.append([x, y, x + w, y + h])
#             mask = self.coco.annToMask(ann)
#             masks.append(mask)

#         results = dict(filename=Path(image_path).name,
#                        ori_shape=image.shape[:2],
#                        image=image,
#                        masks=np.stack(masks).astype(np.float32),
#                        boxes=np.stack(boxes))

#         if self.transform:
#             results = self.transform(results)

#         return results


@DATASETS.register_module()
class HRSIDDataset(CocoDataset):
    METAINFO = {'classes': ('ship'), 'palette': [[220, 20, 60]]}


@DATASETS.register_module()
class WHUBuidlingDataset(CocoDataset):
    METAINFO = {'classes': ('building'), 'palette': [[255, 255, 255]]}
