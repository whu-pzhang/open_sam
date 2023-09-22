import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

from mmcv.transforms import Resize
from mmdet.datasets.transforms import LoadAnnotations

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


def vis_dataset():
    # dataset
    dataset_type = 'mmseg.LoveDADataset'
    data_root = 'data/loveDA/'

    train_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(type='mmseg.LoadAnnotations'),
        # dict(type='mmseg.RandomResize',
        #      scale=(2048, 512),
        #      ratio_range=(0.5, 2.0),
        #      keep_ratio=True),
        # dict(type='mmseg.RandomCrop', crop_size=(512, 512),
        #      cat_max_ratio=0.75),
        # dict(type='mmseg.RandomFlip', prob=0.5),
        # dict(type='mmseg.PhotoMetricDistortion'),
        dict(type='ResizeLongestEdge', scale=1024),
        dict(type='GenerateSAMPrompt', max_instances=15,
             points_per_instance=2),
        # dict(type='ResizeLongestSide', target_length=1024),
        # dict(type='PackSamInputs'),
    ]

    dataset = dict(
        type=dataset_type,
        data_root=data_root,
        img_suffix='.png',
        seg_map_suffix='.png',
        data_prefix=dict(img_path='img_dir/train',
                         seg_map_path='ann_dir/train'),
        #    ann_file='../train.txt',
        pipeline=train_pipeline)

    ds = DATASETS.build(dataset)

    sample = ds[100]
    # print(sample['data_samples'].gt_instances)
    # quit()

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(sample['img'])
    for idx, mask in enumerate(sample['gt_masks']):
        show_mask(mask, ax, random_color=True)
    for box in sample['boxes']:
        show_box(box, ax)

    point_coords = sample['point_coords']  # BxNx2
    point_labels = np.ones(point_coords.shape[:2], dtype=np.uint8)  # BxN
    show_points(point_coords, point_labels, ax, marker_size=200)

    # ax.axis('off')
    plt.tight_layout()
    plt.savefig('junk.jpg')
    plt.show()


def vis_det_dataset():
    dataset_type = 'HRSIDDataset'
    data_root = 'data/HRSID_JPG/'

    train_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(type='mmdet.LoadAnnotations', with_bbox=True, with_mask=True),
        dict(type='mmdet.Resize', scale=(1333, 800), keep_ratio=True),
        dict(type='mmdet.RandomFlip', prob=0.5),
        # dict(type='PackDetInputs')
    ]

    dataset = dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/train2017.json',
        data_prefix=dict(img='JPEGImages/'),
        filter_cfg=dict(filter_empty_gt=True),
        pipeline=train_pipeline,
    )

    ds = DATASETS.build(dataset)

    sample = ds[0]
    for k, v in sample.items():
        print(f'{k}: {v}')


if __name__ == '__main__':
    # vis_dataset()
    vis_det_dataset()
