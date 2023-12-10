from numbers import Number
from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from mmengine.utils import is_seq_of
from mmengine.model import BaseDataPreprocessor, ImgDataPreprocessor
from mmengine.structures import InstanceData

from open_sam.registry import MODELS
from ..datasets import SamDataSample
from open_sam.utils import stack_batch


@MODELS.register_module()
class SamDataPreprocessor(BaseDataPreprocessor):
    '''Image pre-processor for SAM.

    It provides the data pre-processing as follows

    - Collate and move data to the target device.
    - Pad inputs to the maximum size of current batch with defined
      ``pad_value``. The padding size can be divisible by a defined
      ``pad_size_divisor``
    - Stack inputs to batch_inputs.
    - Convert inputs from bgr to rgb if the shape of input is (3, H, W).
    - Normalize image with defined std and mean.
    
    '''

    def __init__(self,
                 mean: Sequence[Number] = None,
                 std: Sequence[Number] = None,
                 size: Optional[tuple] = None,
                 pad_val: Number = 0,
                 mask_pad_val: Number = 255,
                 bgr_to_rgb: bool = False,
                 rgb_to_bgr: bool = False,
                 test_cfg: dict = None):
        super().__init__()
        self.size = size
        self.pad_val = pad_val
        self.mask_pad_val = mask_pad_val

        assert not (bgr_to_rgb and rgb_to_bgr), (
            '`bgr2rgb` and `rgb2bgr` cannot be set to True at the same time')
        self.channel_conversion = rgb_to_bgr or bgr_to_rgb

        if mean is not None:
            assert std is not None, 'To enable the normalization in ' \
                                    'preprocessing, please specify both ' \
                                    '`mean` and `std`.'
            # Enable the normalization in preprocessing.
            self._enable_normalize = True
            self.register_buffer('mean',
                                 torch.tensor(mean).view(-1, 1, 1), False)
            self.register_buffer('std',
                                 torch.tensor(std).view(-1, 1, 1), False)
        else:
            self._enable_normalize = False

        # Support different padding methods in testing
        self.test_cfg = test_cfg

    def forward(self, data: dict, training: bool = False):
        """Perform normalization,padding and bgr2rgb conversion based on
        ``BaseDataPreprocessor``.

        Args:
            data (dict): Data sampled from dataloader.
            training (bool): Whether to enable training time augmentation.

        Returns:
            dict: Data in the same format as the model input.
        """
        data = self.cast_data(data)  # type: ignore
        inputs = data['inputs']
        data_samples = data.get('data_samples', None)

        images = inputs['image']
        # TODO: whether normalize should be after stack_batch
        if self.channel_conversion and images[0].size(0) == 3:
            images = [_image[[2, 1, 0], ...] for _image in images]

        images = [_image.float() for _image in images]
        if self._enable_normalize:
            images = [(_image - self.mean) / self.std for _image in images]
        inputs['image'] = images

        if training:
            assert data_samples is not None, ('During training, ',
                                              '`data_samples` must be define.')
            inputs, data_samples = stack_batch(inputs=inputs,
                                               data_samples=data_samples,
                                               size=self.size,
                                               pad_val=self.pad_val,
                                               mask_pad_val=self.mask_pad_val)
        else:
            img_size = inputs[0].shape[1:]
            assert all(input_.shape[1:] == img_size for input_ in inputs),  \
                'The image size in a batch should be the same.'
            # pad images when testing
            if self.test_cfg:
                inputs, padded_samples = stack_batch(
                    inputs=inputs,
                    size=self.test_cfg.get('size', None),
                    pad_val=self.pad_val,
                    mask_pad_val=self.mask_pad_val)
                for data_sample, pad_info in zip(data_samples, padded_samples):
                    data_sample.set_metainfo({**pad_info})
            else:
                inputs = torch.stack(inputs, dim=0)

        # for k, v in inputs.items():
        #     print(f'{k} = {v.shape}')

        return dict(inputs=inputs, data_samples=data_samples)
