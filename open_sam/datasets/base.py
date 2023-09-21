import copy
import os.path as osp
from typing import Callable, Dict, List, Optional, Sequence, Union

import mmengine
import mmengine.fileio as fileio
import numpy as np
from mmseg.datasets import BaseSegDataset

from open_sam.registry import DATASETS


@DATASETS.register_module()
class SegDataset(BaseSegDataset):
    METAINFO = dict(classes=('bg', 'fg'), palette=[[0, 0, 0], [255, 255, 255]])
