import argparse
import glob
import os.path as osp
from pathlib import Path

import mmcv
import numpy as np
import pycocotools.mask as maskUtils
from mmengine.fileio import dump
from mmengine.utils import (Timer, mkdir_or_exist, track_parallel_progress,
                            track_progress)


def collect_annotations(files, nproc=1):
    print('Loading annotation images')
    if nproc > 1:
        images = track_parallel_progress()


def load_img_info(files):
    img_file, segm_file = files

    segm_map = mmcv.imread(segm_file, 'unchanged')

    unique_class_ids = np.unique(segm_map)

    anno_info = []

    for cls_id in unique_class_ids:

        mask = np.asarray(segm_map == cls_id, dtype=np.uint8)
