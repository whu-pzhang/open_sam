import argparse
import os
import os.path as osp
from pathlib import Path
import json
from functools import partial

import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from pycocotools.coco import COCO
import pycocotools.mask as maskUtils
from mmengine.dataset import DefaultSampler, worker_init_fn
from mmengine.dist import (collect_results, get_dist_info, get_rank, init_dist,
                           is_distributed)
from mmengine.utils import ProgressBar
from mmdet.datasets import CocoDataset

from open_sam import SamPredictor, sam_model_registry


def parse_args():
    parser = argparse.ArgumentParser('Detect-Segment-Anything Demo',
                                     add_help=True)
    parser.add_argument('data_root', type=str)
    parser.add_argument('--ann-file',
                        type=str,
                        default='annotations/whu-building_val.json')
    parser.add_argument('--img-dir', type=str, default='val/image')
    # parser.add_argument('--prompt-type',
    #                     type=str,
    #                     default='box',
    #                     choices=['point', 'box', 'mask'])
    parser.add_argument('--sam-type',
                        type=str,
                        default='vit_t',
                        choices=['vit_h', 'vit_l', 'vit_b', 'vit_t'],
                        help='sam type')
    parser.add_argument('--sam-weight',
                        type=str,
                        default=None,
                        help='path to checkpoint file')
    parser.add_argument('--device',
                        default='cuda',
                        help='Device used for inference')

    parser.add_argument('--out-dir',
                        '-o',
                        type=str,
                        default='outputs',
                        help='output directory')

    parser.add_argument('--num-worker', '-n', type=int, default=2)
    args = parser.parse_args()

    return args


class SimpleDataset(Dataset):

    def __init__(self, img_ids):
        self.img_ids = img_ids

    def __getitem__(self, item):
        return self.img_ids[item]

    def __len__(self):
        return len(self.img_ids)


def sam_boxes_predict(predictor, input_bboxes, img_hw, batch_size=32):
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

        batch_masks, _, _ = predictor.predict_torch(point_coords=None,
                                                    point_labels=None,
                                                    boxes=transformed_boxes,
                                                    multimask_output=False,
                                                    return_logits=False)

        batch_masks = batch_masks.squeeze(1)
        masks.extend([*batch_masks])

    masks = torch.stack(masks, dim=0).cpu().numpy()
    return masks


def fake_collate(x):
    return x


def main():

    args = parse_args()

    coco = COCO(osp.join(args.data_root, args.ann_file))
    # coco_dataset = SimpleDataset(coco.getImgIds())

    coco_dataset = CocoDataset(data_root=args.data_root,
                               ann_file=args.ann_file,
                               metainfo=dict(classes=('building', )),
                               data_prefix=dict(img=args.img_dir),
                               pipeline=[
                                   dict(type='LoadImageFromFile'),
                                   dict(type='mmdet.LoadAnnotations',
                                        with_bbox=True,
                                        with_mask=True)
                               ])

    sampler = DefaultSampler(coco_dataset, shuffle=True)
    init_fn = partial(worker_init_fn,
                      num_workers=args.num_worker,
                      rank=get_rank(),
                      seed=0,
                      disable_subprocess_warning=True)
    data_loader = DataLoader(
        dataset=coco_dataset,
        sampler=sampler,
        collate_fn=fake_collate,
        worker_init_fn=init_fn,
        batch_size=1,
        num_workers=args.num_worker,
        persistent_workers=False if args.num_worker == 0 else True,
        drop_last=False)

    name2id = {}
    for i, name in enumerate(coco_dataset.metainfo['classes']):
        name2id[name] = i

    # build sam
    sam = sam_model_registry[args.sam_type](checkpoint=args.sam_weight)
    sam_predictor = SamPredictor(sam.to(args.device))

    if get_rank() == 0:
        progress_bar = ProgressBar(len(data_loader))

    part_json_data = []

    for i, data in enumerate(data_loader):
        new_json_data = dict(annotation=[])
        # img_id = data[0]
        # raw_img_info = coco.loadImgs(img_id)[0]
        # ann_ids = coco.getAnnIds(imgIds=[img_id])
        # raw_ann_info = coco.loadAnns(ann_ids)

        # file_name = raw_img_info['file_name']
        # image_path = osp.join(args.data_root, args.img_dir, file_name)
        # print(raw_ann_info)

        sample = data[0]
        img_id = sample['img_id']
        img = sample['img']
        gt_bboxes = sample['gt_bboxes'].numpy()
        # gt_masks = sample['gt_masks']
        img_hw = sample['img_shape']

        new_json_data['image'] = dict(id=img_id,
                                      file_name=osp.basename(
                                          sample['img_path']),
                                      height=sample['height'],
                                      width=sample['width'])

        if get_rank() == 0:
            progress_bar.update()

        if len(gt_bboxes) == 0:
            part_json_data.append(new_json_data)
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        sam_predictor.set_image(img)

        masks = sam_boxes_predict(sam_predictor, gt_bboxes, img_hw)

        for i in range(len(gt_bboxes)):
            label_id = sample['gt_bboxes_labels'][i]
            score = 0.5
            bbox = gt_bboxes[i].tolist()
            coco_bbox = [
                bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]
            ]
            mask = masks[i]
            encode_mask = maskUtils.encode(
                np.array(mask[:, :, np.newaxis], order='F', dtype='uint8'))[0]
            encode_mask['counts'] = encode_mask['counts'].decode()

            anno = dict(image_id=int(img_id),
                        bbox=coco_bbox,
                        score=float(score),
                        iscrowd=0,
                        category_id=int(label_id),
                        segmentation=encode_mask,
                        area=maskUtils.area(encode_mask))

            new_json_data['annotation'].append(anno)

        part_json_data.append(new_json_data)

    all_json_results = collect_results(part_json_data, len(coco_dataset),
                                       'cpu')

    if get_rank() == 0:
        new_json_data = {
            'info': coco.dataset.get('info', []),
            'licenses': coco.dataset.get('licenses', []),
            'categories': coco.dataset['categories'],
            'images':
            [json_results['image'] for json_results in all_json_results]
        }

        annotations = []
        annotation_id = 1
        for annotation in all_json_results:
            annotation = annotation['annotation']
            for ann in annotation:
                ann['id'] = annotation_id
                annotation_id += 1
                annotations.append(ann)

        if len(annotations) > 0:
            new_json_data['annotations'] = annotations

        output_json_name = Path(args.ann_file).stem + '_pred.json'
        output_name = osp.join(args.out_dir, output_json_name)
        os.makedirs(os.path.dirname(output_name), exist_ok=True)

        with open(output_name, 'w') as f:
            json.dump(new_json_data, f)


if __name__ == '__main__':
    main()
