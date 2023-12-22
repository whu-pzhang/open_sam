from typing import Optional
from functools import partial

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

from mmcv.transforms import Resize
from mmdet.datasets.transforms import LoadAnnotations
from mmdet.structures.mask import BitmapMasks
from mmengine.dataset import DefaultSampler, worker_init_fn
from mmengine.dist import (collect_results, get_dist_info, get_rank, init_dist,
                           is_distributed)
from mmengine.config import Config
from torch.utils.data import DataLoader, Dataset

from open_sam.registry import MODELS, DATASETS, TRANSFORMS
from open_sam.utils import register_all_modules

register_all_modules()


def show_mask(mask, ax, random_color=False, alpha=0.8):
    if random_color:
        color = np.concatenate(
            [np.random.random(3), np.array([alpha])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, alpha])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0],
               pos_points[:, 1],
               color='green',
               marker='*',
               s=marker_size,
               edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0],
               neg_points[:, 1],
               color='red',
               marker='*',
               s=marker_size,
               edgecolor='white',
               linewidth=1.25)


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


def vis_seg_dataset():
    # dataset
    # dataset_type = 'mmseg.LoveDADataset'
    # data_root = 'data/loveDA/'
    dataset_type = 'SegDataset'
    data_root = 'data/whu-building/cropped_aerial_data'

    train_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(type='mmseg.LoadAnnotations'),
        dict(type='mmseg.RandomResize',
             scale=(2048, 512),
             ratio_range=(0.5, 2.0),
             keep_ratio=True),
        dict(type='mmseg.RandomCrop', crop_size=(512, 512),
             cat_max_ratio=0.75),
        # dict(type='mmseg.RandomFlip', prob=0.5),
        # dict(type='mmseg.PhotoMetricDistortion'),
        dict(type='ResizeLongestEdge', scale=800),
        dict(type='mmdet.Pad', size=(800, 800), pad_val=0),
        dict(type='GenerateSAMPrompt',
             max_instances_per_classes=10,
             points_per_instance=2,
             ignore_values=[0, 255]),
        # dict(type='ResizeLongestSide', target_length=1024),
        dict(type='PackSamInputs'),
    ]

    dataset = dict(
        type=dataset_type,
        data_root=data_root,
        # img_suffix='.png',
        # seg_map_suffix='.png',
        # data_prefix=dict(img_path='img_dir/train',
        #                  seg_map_path='ann_dir/train'),
        #
        img_suffix='.tif',
        seg_map_suffix='.tif',
        data_prefix=dict(img_path='train/image', seg_map_path='train/label'),
        ann_file='../train.txt',
        pipeline=train_pipeline)

    ds = DATASETS.build(dataset)

    sample = ds[10]

    data_samples = sample['data_samples']
    print(data_samples)

    quit()

    for sample in ds:
        # print(sample['data_samples'].gt_instances)
        # quit()

        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.imshow(sample['img'][..., ::-1])
        for idx, mask in enumerate(sample['gt_masks']):
            show_mask(mask, ax, random_color=True)
        for box in sample['boxes']:
            show_box(box, ax)

        # point_coords = sample['point_coords']  # BxNx2
        # point_labels = np.ones(point_coords.shape[:2], dtype=np.uint8)  # BxN
        # show_points(point_coords, point_labels, ax, marker_size=200)

        # ax.axis('off')
        plt.tight_layout()
        # plt.savefig('junk.jpg')
        plt.show()


def vis_coco_dataset():
    # dataset_type = 'HRSIDDataset'
    # data_root = 'data/HRSID_JPG/'
    dataset_type = 'mmdet.CocoDataset'
    # whu-building
    data_root = 'data/whu-building/cropped_aerial_data'
    metainfo = dict(classes=('background', 'building'),
                    palette=([0, 0, 0], [255, 255, 255]))

    # nwpu
    # data_root = 'D:/datasets/02-ObjectDet/nwpu'
    # metainfo = dict(classes=('airplane', 'storage tank', 'baseball diamond',
    #                          'tennis court', 'basketball court',
    #                          'ground track field', 'vehicle', 'harbor',
    #                          'bridge', 'ship'), )
    # # loveda
    # data_root = 'data/loveDA'
    # metainfo = dict(classes=('background', 'building', 'road', 'water',
    #                          'barren', 'forest', 'agricultural'),
    #                 palette=[[255, 255, 255], [255, 0, 0], [255, 255, 0],
    #                          [0, 0, 255], [159, 129, 183], [0, 255, 0],
    #                          [255, 195, 128]])

    # iSAID
    # dataset_type = 'mmdet.iSAIDDataset'
    # data_root = 'data/iSAID_patches'
    # metainfo = dict(classes=('ship', 'store_tank', 'baseball_diamond',
    #                          'tennis_court', 'basketball_court',
    #                          'Ground_Track_Field', 'Bridge', 'Large_Vehicle',
    #                          'Small_Vehicle', 'Helicopter', 'Swimming_pool',
    #                          'Roundabout', 'Soccer_ball_field', 'plane',
    #                          'Harbor'),
    #                 palette=[(0, 0, 63), (0, 63, 63), (0, 63, 0), (0, 63, 127),
    #                          (0, 63, 191), (0, 63, 255), (0, 127, 63),
    #                          (0, 127, 127), (0, 0, 127), (0, 0, 191),
    #                          (0, 0, 255), (0, 191, 127), (0, 127, 191),
    #                          (0, 127, 255), (0, 100, 155)])

    train_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(type='mmdet.LoadAnnotations', with_bbox=True, with_mask=True),
        dict(type='ResizeLongestEdge', scale=1024),
        dict(
            type='GenerateSAMPrompt',
            prompt_type=['point', 'boxes'],
            # prompt_type='boxes',
            max_instances_per_classes=10,
            points_per_instance=1,
            noise_cfg=None,
        ),
        # dict(type='PackSamInputs')
    ]

    dataset = dict(
        type=dataset_type,
        data_root=data_root,
        # ann_file='annotations/train2017.json',
        # data_prefix=dict(img='JPEGImages'),
        metainfo=metainfo,
        ann_file='annotations/whu-building_train.json',
        data_prefix=dict(img='train/image'),
        # ann_file='annotations/train.json',
        # data_prefix=dict(img='images'),
        # ann_file='annotations/loveda_train.json',
        # data_prefix=dict(img='img_dir/train'),
        # ann_file='train/instancesonly_filtered_train.json',
        # data_prefix=dict(img='train/images/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
    )
    coco_dataset = DATASETS.build(dataset)

    sample = coco_dataset[10]

    print(sample.keys())
    quit()

    for sample in coco_dataset:
        # print(sample)
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.imshow(sample['img'][..., ::-1])

        if sample.get('gt_masks', None) is not None:
            for idx, mask in enumerate(sample['gt_masks']):
                show_mask(mask, ax, random_color=True)

        if sample.get('boxes', None) is not None:
            for box in sample['boxes']:
                show_box(box, ax)

        if sample.get('point_coords', None) is not None:
            point_coords = sample['point_coords']  # BxNx2
            point_labels = sample['point_labels']  # BxN
            show_points(point_coords, point_labels, ax, marker_size=100)

        # ax.axis('off')
        plt.tight_layout()
        # plt.savefig('junk.jpg')
        plt.show()


def test_dataloader():
    from mmengine.runner import Runner
    from mmengine.utils import ProgressBar
    from pathlib import Path

    cfg = Config.fromfile('configs/_base_/datasets/isaid_coco.py')
    dataloader_cfg = cfg.train_dataloader
    dataloader_cfg.sampler.shuffle = False

    data_loader = Runner.build_dataloader(dataloader_cfg)

    pbar = ProgressBar(task_num=len(data_loader))

    for data_batch in data_loader:
        inputs = data_batch['inputs']
        data_samples = data_batch['data_samples']

        filenames = [Path(d.img_path).name for d in data_samples]

        # for data_sample in data_samples:
        #     print(data_sample)
        #     gt_masks = data_sample.gt_instances.masks
        #     print(gt_masks.shape)
        #     print(data_sample.prompt_instances)
        #     break
        pbar.update()

        continue


def test_sam_dataset():

    train_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
        dict(type='mmdet.Resize', scale=(2048, 1024), keep_ratio=True),
        dict(type='mmdet.RandomFlip', prob=0.5),
        # dict(type='ResizeLongestEdge', scale=1024),
        dict(
            type='GenerateSAMPrompt',
            prompt_type=['point', 'bbox'],
            # prompt_type='boxes',
            max_instances_per_classes=5,
            points_per_instance=2,
            noise_cfg=None,
        ),
        dict(type='PackSamInputs')
    ]
    sam_dataset_cfg = dict(type='SamDataset',
                           data_root='data/rs_sam_data/train',
                           indices=range(1000, 1010),
                           filter_cfg=dict(min_size=32),
                           pipeline=train_pipeline)

    dataset = DATASETS.build(sam_dataset_cfg)

    sample = dataset[0]

    for k, v in sample.items():
        print(f'{k} = {v}')
    quit()

    for sample in dataset:
        print(sample['img_path'])
        # print(sample)
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.imshow(sample['img'][..., ::-1])

        if sample.get('gt_masks', None) is not None:
            for idx, mask in enumerate(sample['gt_masks']):
                show_mask(mask, ax, random_color=True)

        if sample.get('boxes', None) is not None:
            for box in sample['boxes'].numpy():
                show_box(box, ax)

        if sample.get('point_coords', None) is not None:
            point_coords = sample['point_coords']  # BxNx2
            point_labels = sample['point_labels']  # BxN
            show_points(point_coords, point_labels, ax, marker_size=100)

        # ax.axis('off')
        plt.tight_layout()
        # plt.savefig('junk.jpg')
        plt.show()


if __name__ == '__main__':
    # vis_seg_dataset()
    # vis_coco_dataset()
    # test_dataloader()
    test_sam_dataset()
