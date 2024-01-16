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
                 ignore_index=[0, 255],
                 num_points=1,
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
        self.num_points = num_points

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
                    dict(cat_id=cat_id, bbox=[x, y, x + w - 1, y + h - 1]))

                if ann['segmentation']:
                    instances.update(mask=self.coco.annToMask(ann))
        else:
            seg_map_path = self.ann_dir / data_info['seg_map']
            seg_map = np.array(Image.open(seg_map_path))

            if self.reduce_zero_label:
                seg_map[seg_map == 0] = 255
                seg_map = seg_map - 1
                seg_map[seg_map == 254] = 255

            instances = mask2instance(seg_map,
                                      ignore_index=self.ignore_index,
                                      num_points=self.num_points)

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
         img_dir='images',
         ann_file='annotations/train.json',
         img_suffix='.jpg',
         mode='coco',
         classes=('background', 'airplane', 'storage tank', 'baseball diamond',
                  'tennis court', 'basketball court', 'ground track field',
                  'vehicle', 'harbor', 'bridge', 'ship'),
         palette=[(0, 0, 0), (0, 0, 63), (0, 63, 63), (0, 63, 0), (0, 63, 127),
                  (0, 63, 191), (0, 63, 255), (0, 127, 63), (0, 127, 127),
                  (0, 0, 127), (0, 0, 191)]),
}


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--model-type',
                        type=str,
                        default='base',
                        choices=['edge', 'tiny', 'base', 'large', 'huge'])
    parser.add_argument('--dataset', type=str, default='loveda')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--prompt',
                        type=str,
                        default='bbox',
                        choices=['point', 'bbox'])
    parser.add_argument('--num-points', type=int, default=1)

    args = parser.parse_args()
    return args


def main():

    args = parse_args()
    print(args)

    data_info = DATASET_INFO[args.dataset]
    dataset = RSDataset(**data_info, num_points=args.num_points)

    img_dir, ann_dir = dataset.img_dir, dataset.ann_dir
    if ann_dir is None:
        ann_dir = dataset.data_root

    out_dir = dataset.data_root / f'sam-{args.model_type}-{args.prompt}-prompt_pred'
    out_dir.mkdir(exist_ok=True, parents=True)
    palette = [i for rgb in data_info['palette'] for i in rgb]

    device = get_device()
    sam = build_sam(arch=args.model_type, checkpoint=args.checkpoint)
    sam = sam.to(device)
    sam_predictor = SamPredictor(sam)

    pbar = tqdm(total=len(dataset))
    for data in dataset:
        filename = data['filename']
        img = data['image']
        cat_ids = data.get('cat_id', None)
        bboxes = data.get('bbox', None)
        point_coords = data.get('point_coords', None)
        point_labels = data.get('point_labels', None)

        pbar.set_description(f'{filename}')

        prompt_input = dict(bboxes=bboxes if args.prompt == 'bbox' else None)
        if args.prompt == 'point':
            prompt_input.update(point_coords=point_coords,
                                point_labels=point_labels)

        if cat_ids is not None:
            sam_predictor.set_image(img)
            pred_masks = sam_batch_predict(sam_predictor, img.shape[:2],
                                           **prompt_input)

            masks = []
            for idx, mask in enumerate(pred_masks):
                mask = mask.cpu().numpy()
                area = np.sum(mask)
                ann = dict(segmentation=mask, area=area, id=cat_ids[idx])
                masks.append(ann)

            sorted_masks = sorted(masks,
                                  key=(lambda x: x['area']),
                                  reverse=True)
            seg_maps = instance2segmap(sorted_masks, img.shape[:2])
        else:
            seg_maps = np.zeros(img.shape[:2], dtype=np.uint8)

        out_file = out_dir / f'{Path(filename).stem}.png'
        # cv2.imwrite(str(out_file), seg_maps)
        seg_maps = Image.fromarray(seg_maps).convert('P')
        seg_maps.putpalette(palette)
        seg_maps.save(out_file)

        pbar.update()


if __name__ == '__main__':
    main()
