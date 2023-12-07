import argparse
import os.path as osp
from pathlib import Path

import cv2
import numpy as np
from pycocotools.coco import COCO
import pycocotools.mask as maskUtils
from mmengine.fileio import dump
from mmengine.utils import mkdir_or_exist, ProgressBar


def cvt_to_sa1b_json(args):
    coco = COCO(args.ann_file)

    print(f'Total number of images：{len(coco.getImgIds())}')
    categories = coco.loadCats(coco.getCatIds())
    category_names = [category['name'] for category in categories]
    print(f'Total number of Categories : {len(category_names)}')
    print('Categories: \n{}\n'.format(', '.join(category_names)))

    if args.exculde_category is None:
        category_ids = []
    else:
        assert set(category_names) > set(args.exculde_category)
        keep_category_names = set(category_names) - set(args.exculde_category)
        category_ids = coco.getCatIds(keep_category_names)

    image_ids = coco.getImgIds(catIds=category_ids)

    pbar = ProgressBar(task_num=len(image_ids))

    image_id = 0
    ann_id = 0

    for i in range(len(image_ids)):
        img_info = coco.loadImgs(image_ids[i])[0]

        annotation_ids = coco.getAnnIds(imgIds=img_info['id'],
                                        catIds=category_ids,
                                        iscrowd=0)
        annotations = coco.loadAnns(annotation_ids)

        anno_data = []
        for anno in annotations:
            bbox = anno['bbox']
            w, h = bbox[2], bbox[3]
            segmentation = anno['segmentation']

            if w < 10 or h < 10:
                continue

            if segmentation == []:
                continue

            ann_dict = dict(id=ann_id, bbox=bbox, area=anno['area'])

            if type(segmentation) == list:
                # polygon
                polys = []
                for seg in segmentation:
                    polys.append(np.array(seg))
                rles = maskUtils.frPyObjects(polys, img_info['height'],
                                             img_info['width'])
                rle = maskUtils.merge(rles)
            else:
                # mask
                if type(segmentation['counts']) == list:
                    rle = maskUtils.frPyObjects([segmentation],
                                                img_info['height'],
                                                img_info['width'])
                else:
                    rle = [segmentation]
            rle['counts'] = rle['counts'].decode()
            ann_dict.update(segmentation=rle)
            anno_data.append(ann_dict)

            ann_id += 1

        if len(anno_data) == 0:
            pbar.update()
            continue

        file_stem = Path(img_info['file_name']).stem
        if args.prefix is not None:
            file_stem = args.prefix + f'_{file_stem}'
        image_data = dict(image_id=image_id,
                          height=img_info['height'],
                          width=img_info['width'],
                          file_name=file_stem + '.jpg')
        image_id += 1

        json_dict = dict(image=image_data, annotations=anno_data)
        out_json = osp.join(args.out_dir, f'{file_stem}.json')
        dump(json_dict, out_json)

        if args.data_root is not None:
            image_path = osp.join(args.data_root, args.img_dir,
                                  img_info['file_name'])
        else:
            image_path = osp.join(args.img_dir, img_info['file_name'])
        image = cv2.imread(image_path)
        out_img = osp.join(args.out_dir, f'{file_stem}.jpg')
        cv2.imwrite(out_img, image)

        pbar.update()

    print(f'\nimage_id = {image_id}\nann_id = {ann_id}')

    # ===


def parse_args():
    parser = argparse.ArgumentParser(
        description=
        'Convert semantic segmentation mask to SA-1B format json annotations')
    parser.add_argument('--data-root', default=None, help='dataset root')
    parser.add_argument('--img-dir', help='The root path of images')
    parser.add_argument('--ann-file', help='The root path of images')
    parser.add_argument('--prefix', default=None, help='output prefix')
    parser.add_argument('--out-dir',
                        type=str,
                        help='The output directory of json')
    parser.add_argument('--img-suffix', default='.jpg', help='image suffix')
    parser.add_argument('--exculde_category',
                        type=str,
                        nargs='+',
                        default=None)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    mkdir_or_exist(args.out_dir)
    cvt_to_sa1b_json(args)


if __name__ == '__main__':
    main()
