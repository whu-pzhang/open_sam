import sys
import os
import os.path as osp
from pathlib import Path
from collections import defaultdict
import argparse

import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from mmengine.utils import ProgressBar

from pycocotools.coco import COCO

from open_sam import build_sam, SamPredictor
'''
SAM 分割精度测试：

1. 利用数据集 GT 生成visual prompt（point 或者 bbox）
2. 利用生成的 prompt 引导 SAM 完成分割
3. 评估分割精度
'''


def contours2bbox(contours):
    bboxes = []
    for contour in contours:
        xmin, ymin = np.min(contour, axis=0)
        xmax, ymax = np.max(contour, axis=0)

        # w, h = xmax - xmin, ymax - ymin
        bboxes.append((xmin, ymin, xmax, ymax))

    return bboxes


def bitmap_to_polygon(bitmap):
    """Convert masks from the form of bitmaps to polygons.

    Args:
        bitmap (ndarray): masks in bitmap representation.

    Return:
        list[ndarray]: the converted mask in polygon representation.
        bool: whether the mask has holes.
    """
    bitmap = np.ascontiguousarray(bitmap).astype(np.uint8)
    # cv2.RETR_CCOMP: retrieves all of the contours and organizes them
    #   into a two-level hierarchy. At the top level, there are external
    #   boundaries of the components. At the second level, there are
    #   boundaries of the holes. If there is another contour inside a hole
    #   of a connected component, it is still put at the top level.
    # cv2.CHAIN_APPROX_NONE: stores absolutely all the contour points.
    outs = cv2.findContours(bitmap, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    contours = outs[-2]
    hierarchy = outs[-1]
    if hierarchy is None:
        return [], False
    # hierarchy[i]: 4 elements, for the indexes of next, previous,
    # parent, or nested contours. If there is no corresponding contour,
    # it will be -1.
    with_hole = (hierarchy.reshape(-1, 4)[:, 3] >= 0).any()
    contours = [c.reshape(-1, 2) for c in contours]
    return contours, with_hole


def mask2bbox(mask, ignore_index=255):
    class_vals = np.unique(mask)
    num_classes = len(class_vals)

    bboxes_per_class = defaultdict(list)
    for v in class_vals:
        if v == ignore_index:
            continue
        cur_bitmap = (mask == v)
        contours, _ = bitmap_to_polygon(cur_bitmap)
        bboxes = contours2bbox(contours)
        bboxes_per_class[v] = bboxes
    return bboxes_per_class


def sam_batch_predict(predictor, input_bboxes, img_hw, batch_size=32):
    num_bboxes = len(input_bboxes)
    num_batches = int(np.ceil(num_bboxes / batch_size))

    masks = []
    input_bboxes_tensor = torch.from_numpy(input_bboxes).to(predictor.device)
    for i in range(num_batches):
        left_index = i * batch_size
        right_index = (i + 1) * batch_size
        if i == num_batches - 1:
            batch_boxes = input_bboxes_tensor[left_index:]
        else:
            batch_boxes = input_bboxes_tensor[left_index:right_index]

        transformed_boxes = predictor.transform.apply_boxes_torch(
            batch_boxes, img_hw)

        # output mask: (B,num_masks,H,W)
        # output iou scores: (B,num_masks)
        # output logits: (B,num_masks,256,256)
        batch_masks, scores, logits = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
            return_logits=False)

        batch_masks = batch_masks.squeeze(1)
        masks.extend([*batch_masks])

    masks = torch.stack(masks, dim=0)
    return masks


def instance2segmap(instances, img_hw):
    seg_maps = np.zeros(img_hw, dtype=np.uint8)
    # for class_val, insts in instances.items():
    #     mask = np.sum(insts, axis=0).astype('bool')
    #     cur_class_mask = (mask * class_val).astype(np.uint8)
    #     seg_maps = np.where(mask, cur_class_mask, seg_maps)
    # return seg_maps

    for ann in instances:
        mask = ann['segmentation'].astype('bool')
        # cur_class_mask = (mask * ann['id']).astype(np.uint8)
        # seg_maps = np.where(mask, cur_class_mask, seg_maps)
        seg_maps[mask] = ann['id']

    return seg_maps


def mask2instance(mask, ignore_index=[255]):
    class_vals = np.unique(mask)

    instances = []
    for val in class_vals:
        if val in ignore_index:
            continue
        cur_class_mask = (mask == val).astype(np.uint8)
        num_objects, labels, stats, centroids = cv2.connectedComponentsWithStats(
            cur_class_mask)
        for i in range(1, num_objects):
            cur_object = (labels == i).astype(np.uint8)
            # masks.append(cur_object)

            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            # boxes.append([x, y, x + w - 1, y + h - 1])
            instances.append(
                dict(cat_id=val,
                     mask=cur_object,
                     bbox=[x, y, x + w - 1, y + h - 1]))

    return instances


class RSDataset(Dataset):

    def __init__(self,
                 data_root,
                 img_dir,
                 img_suffix='.tif',
                 seg_map_suffix='.tif',
                 ann_file=None,
                 ann_dir=None,
                 mode='seg_map',
                 reduce_zero_label=False,
                 ignore_index=[255],
                 **kwargs):

        assert mode in ['coco', 'seg_map']

        self.data_root = Path(data_root)
        self.img_dir = img_dir
        self.ann_file = ann_file
        self.ann_dir = ann_dir
        self.mode = mode
        self.reduce_zero_label = reduce_zero_label
        self.img_suffix = img_suffix
        self.seg_map_suffix = seg_map_suffix
        self.ignore_index = ignore_index

        if self.ann_file and (not osp.isabs(self.ann_file)):
            self.ann_file = self.data_root / self.ann_file
        if not (self.img_dir is None or osp.isabs(self.img_dir)):
            self.img_dir = self.data_root / self.img_dir
        if not (self.ann_dir is None or osp.isabs(self.ann_dir)):
            self.ann_dir = self.data_root / self.ann_dir

        if self.mode == 'coco':
            self.data_infos = self.load_coco_annotations(self.ann_file)
        else:
            self.data_infos = self.load_seg_annotations(self.ann_file)

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, idx):
        data_info = self.data_infos[idx]
        img_path = self.img_dir / data_info['filename']
        image = np.array(Image.open(img_path))

        if self.mode == 'coco':
            img_id = data_info['img_id']
            ann_ids = self.coco.getAnnIds(imgIds=[img_id])
            anns = self.coco.loadAnns(ann_ids)

            instances = []
            for ann in anns:
                x, y, w, h = ann['bbox']
                cat_id = ann['category_id']
                instances.append(
                    dict(cat_id=cat_id,
                         bbox=[x, y, x + w - 1, y + h - 1],
                         mask=self.coco.annToMask(ann)))
        else:
            seg_map_path = self.ann_dir / data_info['seg_map']
            seg_map = np.array(Image.open(seg_map_path))

            if self.reduce_zero_label:
                seg_map[seg_map == 0] = 255
                seg_map = seg_map - 1
                seg_map[seg_map == 254] = 255

            instances = mask2instance(seg_map, ignore_index=self.ignore_index)

        if len(instances) == 0:
            results = {}
        else:
            # LD to DL
            results = {
                k: np.array([d[k] for d in instances])
                for k in instances[0]
            }
        results.update(
            dict(filename=data_info['filename'],
                 image=image,
                 ori_shape=image.shape[:2]))

        return results

    def load_coco_annotations(self, ann_file):
        self.coco = COCO(ann_file)
        img_ids = self.coco.imgs.keys()

        data_infos = []
        for i in img_ids:
            img_info = self.coco.loadImgs([i])[0]
            ann_ids = self.coco.getAnnIds(imgIds=i)
            if len(ann_ids) == 0:
                continue
            data_infos.append(dict(filename=img_info['file_name'], img_id=i))

        return data_infos

    def load_seg_annotations(self, ann_file=None):
        data_infos = []
        if ann_file is not None:
            with open(ann_file, 'r') as fp:
                lines = fp.readlines()
            for line in lines:
                img_name = line.strip()
                img_info = dict(filename=img_name + self.img_suffix)
                img_info['seg_map'] = img_name + self.seg_map_suffix
                data_infos.append(img_info)
        else:
            for f in self.ann_dir.glob(f'*{self.seg_map_suffix}'):
                data_infos.append(
                    dict(filename=f.stem + self.img_suffix, seg_map=f.name))

        return data_infos


DATASET_INFO = {
    'whu-building':
    dict(data_root='data/whu-building/cropped_aerial_data',
         img_dir='val/image',
         ann_dir='val/label',
         reduce_zero_label=False,
         img_suffix='.tif',
         seg_map_suffix='.tif',
         classes=('background', 'building'),
         palette=[[0, 0, 0], [255, 255, 255]]),
    'loveda':
    dict(data_root='data/loveDA',
         img_dir='img_dir/val',
         ann_dir='ann_dir/val',
         reduce_zero_label=True,
         img_suffix='.png',
         seg_map_suffix='.png',
         classes=('background', 'building', 'road', 'water', 'barren',
                  'forest', 'agricultural'),
         palette=[[255, 255, 255], [255, 0, 0], [255, 255, 0], [0, 0, 255],
                  [159, 129, 183], [0, 255, 0], [255, 195, 128]]),
    'potsdam':
    dict(
        data_root='data/Potsdam',
        img_dir='img_dir/val',
        ann_dir='ann_dir/val',
        reduce_zero_label=True,
        img_suffix='.png',
        seg_map_suffix='.png',
        ignore_index=[255],  # with clutter
        classes=('impervious_surface', 'building', 'low_vegetation', 'tree',
                 'car', 'clutter'),
        palette=[[255, 255, 255], [0, 0, 255], [0, 255, 255], [0, 255, 0],
                 [255, 255, 0], [255, 0, 0]]),
    'hrsid':
    dict(data_root='data/HRSID_JPG',
         img_prefix='JPEGImages',
         ann_file='annotations/train2017.json',
         mode='coco'),
    'nwpu':
    dict(data_root='data/nwpu',
         img_prefix='images',
         ann_file='annotations/train.json',
         mode='coco'),
}


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type',
                        type=str,
                        default='base',
                        choices=['tiny', 'base', 'large', 'huge'])
    parser.add_argument('--dataset', type=str, default='loveda')
    parser.add_argument('--checkpoint', type=str, default=None)

    args = parser.parse_args()
    return args


def main():

    args = parse_args()

    data_info = DATASET_INFO[args.dataset]

    dataset = RSDataset(**data_info)

    img_dir, ann_dir = dataset.img_dir, dataset.ann_dir
    if ann_dir is None:
        ann_dir = dataset.data_root

    out_dir = ann_dir.parent / f'sam-{args.model_type}-bbox-prompt_pred'
    out_dir.mkdir(exist_ok=True, parents=True)
    reduce_zero_label = data_info['reduce_zero_label']
    palette = [i for rgb in data_info['palette'] for i in rgb]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sam = build_sam(arch=args.model_type, checkpoint=args.checkpoint)
    sam = sam.to(device)
    sam_predictor = SamPredictor(sam)

    suffix = ('*.jpg', '*.png', '*.tiff', '*.tif')
    img_list = [f for s in suffix for f in img_dir.rglob(s)]
    out_dir.mkdir(exist_ok=True, parents=True)

    pbar = tqdm(total=len(dataset))
    for data in dataset:
        filename = data['filename']
        img = data['image']
        cat_ids = data.get('cat_id', None)
        bboxes = data.get('bbox', None)

        pbar.set_description(f'{filename}')

        if cat_ids is not None:
            #
            sam_predictor.set_image(img)

            masks = []
            for cat_id, bbox in zip(cat_ids, bboxes):
                cur_bbox = np.array(bbox)
                cur_masks = sam_batch_predict(sam_predictor, cur_bbox,
                                              img.shape[:2])
                for idx, mask in enumerate(cur_masks):
                    mask = mask.cpu().numpy()
                    area = np.sum(mask)
                    ann = dict(segmentation=mask, area=area, id=cat_id)
                    masks.append(ann)
            sorted_masks = sorted(masks,
                                  key=(lambda x: x['area']),
                                  reverse=True)
            seg_maps = instance2segmap(sorted_masks, img.shape[:2])
        else:
            seg_maps = np.zeros(img.shape[:2], dtype=np.uint8)

        out_file = out_dir / filename
        # cv2.imwrite(str(out_file), seg_maps)
        seg_maps = Image.fromarray(seg_maps).convert('P')
        seg_maps.putpalette(palette)
        seg_maps.save(out_file)

        pbar.update()


if __name__ == '__main__':
    main()
