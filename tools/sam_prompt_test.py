import os.path as osp
from pathlib import Path
from collections import defaultdict
import argparse

import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from mmengine.device import get_device
from mmengine.runner.checkpoint import load_checkpoint
from pycocotools.coco import COCO

from open_sam import build_sam, SamPredictor
'''
SAM 分割精度测试：

1. 利用数据集 GT 生成visual prompt（point 或者 bbox）
2. 利用生成的 prompt 引导 SAM 完成分割
3. 评估分割精度
'''


def point_sampling(mask, num_points=2):
    fg_coords = np.argwhere(mask > 0)[:, ::-1]
    fg_size = len(fg_coords)

    num_fg = num_points
    fg_indices = np.random.choice(fg_size, size=num_fg, replace=True)
    fg_coords = fg_coords[fg_indices]
    coords = fg_coords
    labels = np.ones(num_fg).astype(int)
    indices = np.random.permutation(num_points)
    return coords[indices], labels[indices]


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


def sam_batch_predict(predictor,
                      img_hw,
                      point_coords=None,
                      point_labels=None,
                      bboxes=None,
                      batch_size=32):

    if bboxes is not None:
        num_prompts = len(bboxes)
        bboxes = torch.from_numpy(bboxes).to(predictor.device)

    if point_coords is not None:
        num_prompts = len(point_coords)
        point_coords = torch.from_numpy(point_coords).to(predictor.device)
        point_labels = torch.from_numpy(point_labels).to(predictor.device)
        # print(point_coords.shape, point_labels.shape)

    num_batches = int(np.ceil(num_prompts / batch_size))
    masks = []
    for i in range(num_batches):
        left_index = i * batch_size
        right_index = (i +
                       1) * batch_size if i < num_batches - 1 else num_prompts

        boxes_torch = None
        coords_torch = None
        labels_torch = None

        if bboxes is not None:
            boxes_torch = bboxes[left_index:right_index]
            boxes_torch = predictor.transform.apply_boxes_torch(
                boxes_torch, img_hw)
        else:
            coords_torch = point_coords[left_index:right_index]
            labels_torch = point_labels[left_index:right_index]
            coords_torch = predictor.transform.apply_coords_torch(
                coords_torch, img_hw)

        # output mask: (B,num_masks,H,W)
        # output iou scores: (B,num_masks)
        # output logits: (B,num_masks,256,256)
        batch_masks, scores, logits = predictor.predict_torch(
            point_coords=coords_torch,
            point_labels=labels_torch,
            boxes=boxes_torch,
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


def mask2instance(mask, ignore_index=[255], num_points=2):
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

            point_coords, point_labels = point_sampling(cur_object,
                                                        num_points=num_points)
            instances.append(
                dict(cat_id=int(val),
                     mask=cur_object,
                     bbox=[x, y, x + w - 1, y + h - 1],
                     point_coords=point_coords,
                     point_labels=point_labels))

    return instances


DATASET_INFO = {
    'whu-building':
    dict(data_root='data/whu-building/cropped_aerial_data',
         img_dir='val/image',
         ann_dir='val/label',
         reduce_zero_label=False,
         img_suffix='.tif',
         seg_map_suffix='.tif',
         classes=('background', 'building'),
         palette=[[0, 0, 0], [255, 255, 255]],
         ignore_index=[0, 255]),
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
                  [159, 129, 183], [0, 255, 0], [255, 195, 128]],
         ignore_index=[0, 255]),
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
    parser.add_argument('config', type=str)
    parser.add_argument('checkpoint', type=str, default=None)
    parser.add_argument('--prompt',
                        type=str,
                        default='bbox',
                        choices=['point', 'bbox'])
    parser.add_argument('--num-points', type=int, default=1)

    args = parser.parse_args()
    return args


import matplotlib.pyplot as plt


def main():

    from mmengine.config import Config
    from mmengine.runner import Runner
    from mmengine.device import get_device
    from open_sam.registry import MODELS

    args = parse_args()

    cfg = Config.fromfile(args.config)

    # model
    cfg.load_from = args.checkpoint
    device = get_device()
    model = MODELS.build(cfg.model)
    load_checkpoint(model, args.checkpoint, strict=True)

    model = model.to(device)
    model.eval()

    # dataset
    data_loader = Runner.build_dataloader(cfg.test_dataloader)

    for data_batch in data_loader:
        inputs = data_batch['inputs']
        data_samples = data_batch['data_samples']

        if not data_samples[0].prompt_instances:
            continue

        data = model.data_preprocessor(data_batch)
        output = model(**data, mode='predict')
        # output = model.test_step(data_batch)

        gt_masks = output[0].gt_instances.masks.cpu().numpy()
        pred_masks = output[0].pred_instances.masks.cpu().numpy()
        boxes = data_samples[0].prompt_instances['boxes'].cpu().numpy()

        for idx, (gt, pred) in enumerate(zip(gt_masks, pred_masks)):

            f, ax = plt.subplots(1, 2)
            ax[0].imshow(gt)
            ax[1].imshow(pred)
            # show_box(boxes[idx], ax[1])

            plt.show()

        break


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0),
                      w,
                      h,
                      edgecolor='yellow',
                      facecolor=(0, 0, 0, 0),
                      lw=2))


if __name__ == '__main__':
    main()
