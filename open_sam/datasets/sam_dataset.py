import os.path as osp
import json
from typing import Callable, Dict, List, Optional, Sequence, Union

from mmengine.dataset import BaseDataset
import mmengine.fileio as fileio

from open_sam.registry import DATASETS


@DATASETS.register_module()
class SamDataset(BaseDataset):

    METAINFO = dict(classes=('bg', 'fg'), palette=([0, 0, 0], [255, 255, 255]))

    def __init__(self,
                 data_prefix: dict = dict(img_path='', json_path=''),
                 **kwargs):
        super().__init__(data_prefix=data_prefix, **kwargs)

    def load_data_list(self) -> List[dict]:
        """Load annotations from an annotation file named as ``self.ann_file``

        Returns:
            List[dict]: A list of annotation.
        """  # noqa: E501
        data_list = []
        ann_dir = self.data_prefix.get('json_path', None)
        for json_file in fileio.list_dir_or_file(dir_path=ann_dir,
                                                 list_dir=False,
                                                 suffix='.json',
                                                 recursive=True):
            json_file = osp.join(ann_dir, json_file)
            with open(json_file, 'r') as f:
                raw_data_info = json.load(f)
            parsed_data_info = self.parse_data_info(raw_data_info)
            data_list.append(parsed_data_info)

        return data_list

    def full_init(self):
        if self._fully_initialized:
            return
        # load data information
        self.data_list = self.load_data_list()

        # filter illegal data, such as data that has no annotations.
        self.data_list = self.filter_data()

        # Get subset data according to indices.
        if self._indices is not None:
            self.data_list = self._get_unserialized_subset(self._indices)

        # serialize data_list
        if self.serialize_data:
            self.data_bytes, self.data_address = self._serialize_data()

        self._fully_initialized = True

    def parse_data_info(self, raw_data_info: dict) -> Union[dict, List[dict]]:
        img_info = raw_data_info['image']
        ann_info = raw_data_info['annotations']

        data_info = {}
        data_info['img_path'] = osp.join(self.data_prefix['img_path'],
                                         img_info['file_name'])
        data_info['img_id'] = img_info['image_id']
        data_info['height'] = img_info['height']
        data_info['width'] = img_info['width']

        instances = []
        for i, ann in enumerate(ann_info):
            instance = {}
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue

            if ann.get('iscrowd', False):
                instance['ignore_flag'] = 1
            else:
                instance['ignore_flag'] = 0

            instance['bbox'] = [x1, y1, x1 + w, y1 + h]
            if ann.get('segmentation', None):
                instance['mask'] = ann['segmentation']
            instances.append(instance)

        data_info['instances'] = instances

        return data_info

    def filter_data(self) -> List[dict]:
        """Filter annotations according to filter_cfg.

        Returns:
            List[dict]: Filtered results.
        """
        if self.test_mode:
            return self.data_list

        if self.filter_cfg is None:
            return self.data_list

        min_size = self.filter_cfg.get('min_size', 0)

        valid_data_infos = []
        for i, data_info in enumerate(self.data_list):
            width = data_info['width']
            height = data_info['height']
            if min(width, height) >= min_size:
                valid_data_infos.append(data_info)

        return valid_data_infos
