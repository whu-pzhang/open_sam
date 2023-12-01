import argparse
import os.path as osp
from pathlib import Path
from functools import partial

from PIL import Image
import cv2
import mmcv
import numpy as np
import pycocotools.mask as maskUtils
from mmengine.fileio import dump, list_from_file
from mmengine.utils import (Timer, mkdir_or_exist, scandir,
                            track_iter_progress, track_parallel_progress,
                            track_progress)


def collect_image_files(img_dir,
                        ann_dir,
                        split=None,
                        img_suffix='.tif',
                        seg_map_suffix='.tif'):
    img_files = []

    if split is None:
        images_generator = scandir(img_dir, suffix=img_suffix, recursive=True)
        for image_path in track_iter_progress(list(images_generator)):
            image_path = Path(img_dir) / image_path
            mask_path = Path(ann_dir) / (image_path.stem + seg_map_suffix)

            img_files.append((image_path, mask_path))
    else:
        img_ids = list_from_file(split)
        for img_id in track_iter_progress(img_ids):
            image_path = Path(img_dir) / f'img_id{img_suffix}'
            mask_path = Path(ann_dir) / f'img_id{seg_map_suffix}'

            img_files.append((image_path, mask_path))

    return img_files


def collect_annotations(data_path,
                        ignore_values,
                        reduce_zero_label=False,
                        nproc=1):
    print('Loading annotation images')

    fn = partial(mask2coco,
                 ignore_values=ignore_values,
                 reduce_zero_label=reduce_zero_label)
    if nproc > 1:
        images = track_parallel_progress(fn, data_path, nproc=nproc)
    else:
        images = track_progress(fn, data_path)

    return images


def mask2coco(data_info, ignore_values=[0, 255], reduce_zero_label=False):
    img_file, mask_file = data_info
    # get image info
    img = Image.open(img_file)
    img_info = dict(file_name=Path(img_file).name,
                    height=img.height,
                    width=img.width)

    sem_seg_map = mmcv.imread(mask_file, flag='unchanged', backend='pillow')
    sem_seg_map = sem_seg_map.astype(np.uint8)
    if reduce_zero_label:
        sem_seg_map[sem_seg_map == 0] = 255
        sem_seg_map = sem_seg_map - 1
        sem_seg_map[sem_seg_map == 254] = 255

    unique_class_ids = np.unique(sem_seg_map)

    ann_info = []

    for val in unique_class_ids:
        if val in ignore_values:
            continue

        mask = (sem_seg_map == val).astype(np.uint8)
        if mask.max() < 1:
            continue

        mask = cv2.morphologyEx(mask,
                                cv2.MORPH_OPEN,
                                kernel=np.ones((3, 3), np.uint8))
        num_objects, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask, connectivity=4)

        for i in range(1, num_objects):
            cur_object = np.asarray(labels == i, dtype=np.uint8, order='F')
            mask_rle = maskUtils.encode(cur_object)
            area = maskUtils.area(mask_rle)
            bbox = maskUtils.toBbox(mask_rle)

            # for json encoding
            mask_rle['counts'] = mask_rle['counts'].decode()
            ann = dict(iscrowd=0,
                       category_id=val,
                       bbox=bbox.tolist(),
                       area=area.tolist(),
                       segmentation=mask_rle)
            ann_info.append(ann)

    img_info['ann_info'] = ann_info

    return img_info


def cvt_to_coco_json(img_infos, classes):
    coco = dict()

    image_id = 0
    ann_id = 0

    coco['images'] = []
    coco['type'] = 'instance'
    coco['categories'] = []
    coco['annotations'] = []
    image_set = set()

    for category_id, name in enumerate(classes):
        category_item = dict()
        category_item['supercategory'] = str('none')
        category_item['id'] = int(category_id)
        category_item['name'] = str(name)
        coco['categories'].append(category_item)

    for img_dict in img_infos:
        file_name = img_dict['file_name']
        assert file_name not in image_set
        image_item = dict()
        image_item['id'] = int(image_id)
        image_item['file_name'] = str(file_name)
        image_item['height'] = int(img_dict['height'])
        image_item['width'] = int(img_dict['width'])
        coco['images'].append(image_item)
        image_set.add(file_name)

        #
        ann_info = img_dict.pop('ann_info')
        for ann_dict in ann_info:
            ann_dict['image_id'] = image_id
            ann_dict['id'] = ann_id
            coco['annotations'].append(ann_dict)

            ann_id += 1

        image_id += 1

    return coco


def parse_args():
    parser = argparse.ArgumentParser(
        description=
        'Convert semantic segmentation mask to coco format annotations')
    parser.add_argument('--img-path', help='The root path of images')
    parser.add_argument('--ann-path', help='The root path of images')
    parser.add_argument('--split', default=None, help='split text')
    parser.add_argument('--classes',
                        type=str,
                        help='The text file name of storage class list')
    parser.add_argument(
        '--out',
        type=str,
        help='The output annotation json file name, The save dir is in the '
        'same directory as img_path')
    parser.add_argument(
        '-e',
        '--exclude-extensions',
        type=str,
        nargs='+',
        help='The suffix of images to be excluded, such as "png" and "bmp"')
    parser.add_argument('--img-suffix', default='.jpg', help='image suffix')
    parser.add_argument('--seg-map-suffix',
                        default='.png',
                        help='semantic segmentation map suffix')
    parser.add_argument('--reduce-zero-label', action='store_true')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    assert args.out.endswith(
        'json'), 'The output file name must be json suffix'

    # 1 load image list info
    img_files = collect_image_files(args.img_path,
                                    args.ann_path,
                                    split=args.split,
                                    img_suffix=args.img_suffix,
                                    seg_map_suffix=args.seg_map_suffix)

    # 2 convert to coco format data
    img_infos = collect_annotations(img_files,
                                    ignore_values=[0, 255],
                                    reduce_zero_label=args.reduce_zero_label,
                                    nproc=4)
    classes = list_from_file(args.classes)
    coco_info = cvt_to_coco_json(img_infos, classes)

    # 3 dump
    save_dir = osp.dirname(args.out)
    mkdir_or_exist(save_dir)
    print(f'save json file: {args.out}')
    dump(coco_info, args.out)


if __name__ == '__main__':
    main()
