import argparse
import os.path as osp
from pathlib import Path
import json

import cv2
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection, PolyCollection
from matplotlib.patches import Polygon
from pycocotools.coco import COCO
import pycocotools.mask as maskUtils
from mmdet.structures.mask import bitmap_to_polygon


def show_sa1b_json(file_stem_list, data_root):
    '''
    
    {
        "image": 
            "image_id":
            "height":
            "width":
            "file_name":

        "annotations": [
            {
                "bbox":
                "area":
                "segmentation":
                "id":
                "predicted_iou":
                "point_coords":
                "crop_box":
                "stability_score"
            }
        ]
    }
    
    '''

    for file_stem in file_stem_list:
        ann_file = osp.join(data_root, f'{file_stem}.json')

        with open(ann_file, 'r') as f:
            anno_info = json.load(f)
        img_info = anno_info['image']

        img_file = osp.join(data_root, img_info['file_name'])
        # image = cv2.imread(img_file)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.array(Image.open(img_file))

        plt.figure()
        plt.imshow(image)
        ax = plt.gca()

        polygons = []
        colors = []
        for ann in anno_info['annotations']:
            c = (np.random.random((1, 3)) * 0.6 + 0.4).tolist()[0]
            bbox = ann['bbox']
            segmentation = ann['segmentation']

            if type(segmentation) == list:
                # polygon
                for seg in segmentation:
                    poly = np.array(seg).reshape((int(len(seg) / 2), 2))
                    polygons.append(Polygon(poly))
                    colors.append(c)
            else:
                # mask
                if type(segmentation['counts']) == list:
                    rle = maskUtils.frPyObjects([segmentation],
                                                img_info['height'],
                                                img_info['width'])
                else:
                    rle = [segmentation]

                m = maskUtils.decode(rle)

                contours, _ = bitmap_to_polygon(m)
                polygons.extend(contours)

                img = np.ones((m.shape[0], m.shape[1], 3))
                color_mask = np.random.random((1, 3)).tolist()[0]
                for i in range(3):
                    img[:, :, i] = color_mask[i]

                ax.imshow(np.dstack((img, m * 0.5)))

        p = PolyCollection(polygons,
                           alpha=0.5,
                           facecolor='none',
                           edgecolors='b',
                           linewidth=2)
        ax.add_collection(p)

        plt.show()


def show_bbox_only(coco, anns, show_label_bbox=True, is_filling=True):
    """Show bounding box of annotations Only."""
    if len(anns) == 0:
        return

    ax = plt.gca()
    ax.set_autoscale_on(False)

    image2color = dict()
    for cat in coco.getCatIds():
        image2color[cat] = (np.random.random((1, 3)) * 0.7 + 0.3).tolist()[0]

    polygons = []
    colors = []

    for ann in anns:
        color = image2color[ann['category_id']]
        bbox_x, bbox_y, bbox_w, bbox_h = ann['bbox']
        poly = [[bbox_x, bbox_y], [bbox_x, bbox_y + bbox_h],
                [bbox_x + bbox_w, bbox_y + bbox_h], [bbox_x + bbox_w, bbox_y]]
        polygons.append(Polygon(np.array(poly).reshape((4, 2))))
        colors.append(color)

        if show_label_bbox:
            label_bbox = dict(facecolor=color)
        else:
            label_bbox = None

        ax.text(bbox_x,
                bbox_y,
                '%s' % (coco.loadCats(ann['category_id'])[0]['name']),
                color='white',
                bbox=label_bbox)

    if is_filling:
        p = PatchCollection(polygons,
                            facecolor=colors,
                            linewidths=0,
                            alpha=0.4)
        ax.add_collection(p)
    p = PatchCollection(polygons,
                        facecolor='none',
                        edgecolors=colors,
                        linewidths=2)
    ax.add_collection(p)


def parse_args():
    parser = argparse.ArgumentParser(description='Show SA-1B format json file')
    parser.add_argument('--data-root', default=None, help='dataset root')
    parser.add_argument('--wait-time',
                        type=float,
                        default=2,
                        help='the interval of show (s)')
    parser.add_argument('--disp-all',
                        action='store_true',
                        help='Whether to display all types of data, '
                        'such as bbox and mask.'
                        ' Default is to display only bbox')
    parser.add_argument('--shuffle',
                        action='store_true',
                        help='Whether to display in disorder')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    file_stem_list = [f.stem for f in Path(args.data_root).glob('*.json')]

    show_sa1b_json(file_stem_list, args.data_root)


if __name__ == '__main__':
    main()
