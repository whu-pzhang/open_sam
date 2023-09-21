import torch

from mmengine.model import BaseDataPreprocessor
from open_sam.registry import MODELS


class SamDataPreprocessor(BaseDataPreprocessor):

    def __init__(self, mean, std):
        self.register_buffer('mean', torch.tensor(mean).view(-1, 1, 1), False)
        self.register_buffer('std', torch.tensor(std).view(-1, 1, 1), False)

    def forward(self, data: dict, training: bool = False):
        data = self.cast_data(data)

        inputs = data['inputs']
