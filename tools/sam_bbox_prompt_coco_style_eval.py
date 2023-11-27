import argparse
import json
import os
import os.path as osp
import warnings
from functools import partial

import cv2
import numpy as np
import pycocotools.mask as mask_util
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from mmengine.config import Config
from mmengine.dataset import DefaultSampler, worker_init_fn
from mmengine.dist import (collect_results, get_dist_info, get_rank, init_dist,
                           is_distributed)
from mmengine.utils import ProgressBar
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from open_sam import build_sam, SamPredictor


class SimpleDataset(Dataset):

    def __init__(self, img_ids):
        self.img_ids = img_ids

    def __getitem__(self, item):
        return self.img_ids[item]

    def __len__(self):
        return len(self.img_ids)


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', default=None, help='dataset root')
    parser.add_argument('--ann-file',
                        type=str,
                        default='annotations/instances_val2017.json')
    parser.add_argument('--img-dir', type=str, default='val2017/')
    parser.add_argument('--sam-type',
                        type=str,
                        default='tiny',
                        choices=['tiny', 'base', 'large', 'huge'])
    # parser.add_argument('--dataset', type=str, default='loveda')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--out-dir',
                        '-o',
                        type=str,
                        default='outputs',
                        help='output directory')
    parser.add_argument('--num-worker', '-n', type=int, default=2)

    args = parser.parse_args()
    return args


def fake_collate(x):
    return x


def build_detector(config, weight):
    pass


def sam_batch_predict(predictor, input_bboxes, img_hw, batch_size=32):
    num_bboxes = len(input_bboxes)
    num_batches = int(np.ceil(num_bboxes / batch_size))

    masks = []
    all_scores = []
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

        batch_masks, scores, logits = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
            return_logits=False)

        batch_masks = batch_masks.squeeze(1)
        masks.extend([*batch_masks])

        all_scores.append(scores)

    masks = torch.stack(masks, dim=0)
    all_scores = torch.cat(all_scores)
    return masks, all_scores


def main():
    import matplotlib.pyplot as plt

    args = parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sam = build_sam(arch=args.sam_type, checkpoint=args.checkpoint)
    sam = sam.to(device)
    sam_predictor = SamPredictor(sam)

    #
    if args.data_root is not None:
        coco = COCO(osp.join(args.data_root, args.ann_file))
    else:
        coco = COCO(args.ann_file)
    coco_dataset = SimpleDataset(coco.getImgIds())

    name2id = {}
    for categories in coco.dataset['categories']:
        name2id[categories['name']] = categories['id']
    if get_rank() == 0:
        print(f'\nCategories: \n{name2id}\n')

    sampler = DefaultSampler(coco_dataset, False)
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

    if get_rank() == 0:
        progress_bar = ProgressBar(len(data_loader))

    part_json_data = []

    if args.data_root is not None:
        img_dir = osp.join(args.data_root, args.img_dir)
    else:
        img_dir = args.img_dir

    for i, data in enumerate(data_loader):
        new_json_data = dict(annotation=[])
        image_id = data[0]
        raw_img_info = coco.loadImgs([image_id])[0]
        raw_img_info['img_id'] = image_id
        new_json_data['image'] = raw_img_info
        img_hw = (raw_img_info['height'], raw_img_info['width'])

        file_name = raw_img_info['file_name']
        image_path = osp.join(img_dir, file_name)

        annotation_ids = coco.getAnnIds(imgIds=image_id, catIds=[], iscrowd=0)
        annotations = coco.loadAnns(annotation_ids)

        if len(annotations) == 0:
            if get_rank() == 0:
                progress_bar.update()
            continue

        coco_bboxes = np.array([ann['bbox'] for ann in annotations])
        bboxes = coco_bboxes.copy()
        bboxes[:, 2] = bboxes[:, 0] + bboxes[:, 2]
        bboxes[:, 3] = bboxes[:, 1] + bboxes[:, 3]

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # sam predict
        sam_predictor.set_image(image)
        masks, scores = sam_batch_predict(sam_predictor, bboxes, img_hw)
        masks = masks.detach().cpu().numpy()
        scores = scores.detach().cpu().numpy()

        for idx, anno in enumerate(annotations):
            coco_bbox = anno['bbox']
            category_id = anno['category_id']
            annotation = dict(image_id=image_id,
                              bbox=coco_bboxes[idx],
                              score=float(scores[idx]),
                              iscrowd=0,
                              category_id=category_id,
                              area=coco_bbox[2] * coco_bbox[3])

            mask = masks[idx]
            encode_mask = mask_util.encode(
                np.array(mask[:, :, np.newaxis], order='F', dtype='uint8'))[0]
            encode_mask['counts'] = encode_mask['counts'].decode()
            annotation['segmentation'] = encode_mask

            new_json_data['annotation'].append(annotation)

        part_json_data.append(new_json_data)

        if get_rank() == 0:
            progress_bar.update()

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
                # ann['id'] = annotation_id
                annotation_id += 1
                annotations.append(ann)

        if len(annotations) > 0:
            new_json_data['annotations'] = annotations

        output_json_name = args.ann_file[:-5] + '_pred.json'
        output_name = os.path.join(args.out_dir, output_json_name)
        os.makedirs(os.path.dirname(output_name), exist_ok=True)

        print(f'\noutput json: {output_name}\n')
        with open(output_name, 'w') as f:
            json.dump(new_json_data, f)

        if len(coco.dataset['annotations']) > 0:
            cocoDt = COCO(output_name)

            metrics = ['bbox', 'segm']

            for metric in metrics:
                coco_eval = COCOeval(coco, cocoDt, iouType=metric)
                coco_eval.evaluate()
                coco_eval.accumulate()
                coco_eval.summarize()
        else:
            warnings.warn("No gt label, can't evaluate")


if __name__ == '__main__':
    main()
