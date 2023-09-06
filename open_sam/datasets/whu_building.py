import os
from pathlib import Path
import random

import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.utils.data.dataloader import default_collate

from segment_anything.utils.transforms import ResizeLongestSide
from pycocotools.coco import COCO

from mmseg.registry import DATASETS

prompt_type = ['point', 'box', 'mask']


@DATASETS.register_module()
class WHUBuildingDataset(Dataset):

    def __init__(self,
                 data_root,
                 split='train',
                 ann_file='train.txt',
                 transform=None,
                 ignore_index=[0, 255],
                 max_objects=10,
                 points_per_instance=1):
        self.data_root = data_root
        self.transform = transform
        self.ignore_index = ignore_index

        self.img_dir = Path(data_root).joinpath(split, 'image')
        self.ann_dir = Path(data_root).joinpath(split, 'label')
        # self.image_list = [f.stem for f in self.img_dir.glob('*.tif')]

        self.ann_file = ann_file
        # self.image_ids = self.get_info()
        self.img_infos = self.load_anns()

        self.prompt_types = ['point', 'box']

        self.max_objects = max_objects
        self.point_per_instance = points_per_instance

    def load_data_list(self):
        data_list = []

    def load_anns(self):
        with open(self.ann_file, 'r') as fp:
            lines = fp.readlines()

        img_infos = []
        for line in lines:
            img_name = line.strip()
            img_info = dict(filename=img_name + '.tif')
            img_info['seg_map'] = img_name + '.tif'

            img_infos.append(img_info)

        return img_infos

    def __len__(self):
        return len(self.img_infos)

    def __getitem__(self, idx):
        img_path = self.img_dir / self.img_infos[idx]['filename']
        gt_path = self.ann_dir / self.img_infos[idx]['seg_map']

        image = np.array(Image.open(img_path))
        label = np.array(Image.open(gt_path)).astype(np.uint8)
        img_h, img_w = image.shape[:2]

        masks = []
        boxes = []
        point_coords = []
        points_per_instance = self.point_per_instance

        class_vals, counts = np.unique(label, return_counts=True)
        for val in class_vals:
            if val in self.ignore_index:
                continue
            cur_class_mask = (label == val).astype(np.uint8)
            num_objects, labels, stats, centroids = cv2.connectedComponentsWithStats(
                cur_class_mask)

            try:
                object_idxs = random.sample(range(1, num_objects),
                                            k=self.max_objects)
            except:
                object_idxs = range(1, num_objects)

            for i in object_idxs:
                cur_object = (labels == i)
                masks.append(cur_object.astype(np.uint8))

                x = stats[i, cv2.CC_STAT_LEFT]
                y = stats[i, cv2.CC_STAT_TOP]
                w = stats[i, cv2.CC_STAT_WIDTH]
                h = stats[i, cv2.CC_STAT_HEIGHT]
                boxes.append([x, y, x + w - 1, y + h - 1])

                # sample points
                y, x = torch.meshgrid(torch.arange(0, img_h),
                                      torch.arange(0, img_w),
                                      indexing='ij')
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

        results = dict(
            filename=img_path.name,
            img=image,
            img_shape=image.shape[:2],
            ori_shape=image.shape[:2],
            gt_masks=np.stack(masks).astype(np.uint8),
            prompt_type=random.choice(self.prompt_types),
            point_coords=np.stack(point_coords),
            boxes=np.stack(boxes),
        )

        if self.transform:
            results = self.transform(results)

        return results


class COCODataset(Dataset):

    def __init__(self, root_dir, annotation_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.coco = COCO(annotation_file)
        self.image_ids = list(self.coco.imgs.keys())

        # Filter out image_ids without any annotations
        self.image_ids = [
            image_id for image_id in self.image_ids
            if len(self.coco.getAnnIds(imgIds=image_id)) > 0
        ]

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_info = self.coco.loadImgs(image_id)[0]
        image_path = os.path.join(self.root_dir, image_info['file_name'])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        anns = self.coco.loadAnns(ann_ids)
        boxes = []
        masks = []

        for ann in anns:
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])
            mask = self.coco.annToMask(ann)
            masks.append(mask)

        results = dict(filename=Path(image_path).name,
                       ori_shape=image.shape[:2],
                       image=image,
                       masks=np.stack(masks).astype(np.float32),
                       boxes=np.stack(boxes))

        if self.transform:
            results = self.transform(results)

        return results


class ResizeAndPad:

    def __init__(self, target_size):
        self.target_size = target_size
        self.transform = ResizeLongestSide(target_length=target_size)
        # self.to_tensor = T.ToTensor()

    def __call__(self, results):
        image = results['image']
        gt_masks = results['gt_masks']
        boxes = results['boxes']

        ori_h, ori_w = image.shape[:2]
        image = self.transform.apply_image(image)
        gt_masks = [
            torch.as_tensor(self.transform.apply_image(mask))
            for mask in gt_masks
        ]
        # image = self.to_tensor(image)
        # convert to tensor
        image = torch.from_numpy(image.transpose(2, 0, 1)).contiguous().float()

        _, h, w = image.shape
        max_dim = max(h, w)
        pad_w = (max_dim - w) // 2
        pad_h = (max_dim - h) // 2

        padding = (pad_w, pad_h, max_dim - w - pad_w, max_dim - h - pad_h)
        image = T.Pad(padding)(image)
        gt_masks = [T.Pad(padding)(mask) for mask in gt_masks]

        # Adjust bounding boxes
        boxes = self.transform.apply_boxes(boxes, (ori_h, ori_w))
        boxes = [[
            box[0] + pad_w, box[1] + pad_h, box[2] + pad_w, box[3] + pad_h
        ] for box in boxes]

        boxes = np.stack(boxes, axis=0)
        gt_masks = np.stack(gt_masks, axis=0)

        results['image'] = image
        results['gt_masks'] = gt_masks
        results['boxes'] = boxes

        return results
