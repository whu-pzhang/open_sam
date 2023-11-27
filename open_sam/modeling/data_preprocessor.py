import torch

from mmengine.model import BaseDataPreprocessor
from open_sam.registry import MODELS


@MODELS.register_module()
class SamDataPreprocessor(BaseDataPreprocessor):
    '''Image pre-processor for SAM.
    
    '''

    def __init__(self,
                 mean=None,
                 std=None,
                 size=None,
                 size_divisor=None,
                 pad_val=0,
                 bgr_to_rgb=False,
                 rgb_to_bgr=False):
        super().__init__()
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val

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

    def forward(self, data: dict, training: bool = False):
        data = self.cast_data(data)

        inputs = data['inputs']
        data_samples = data.get('data_samples', None)
        # TODO: whether normalize should be after stack_batch
        if self.channel_conversion and inputs[0].size(0) == 3:
            inputs = [_input[[2, 1, 0], ...] for _input in inputs]

        inputs = [_input.float() for _input in inputs]
        if self._enable_normalize:
            inputs = [(_input - self.mean) / self.std for _input in inputs]

        if training:
            assert data_samples is not None, ('During training, ',
                                              '`data_samples` must be define.')
            inputs, data_samples = stack_batch(inputs=inputs,
                                               data_samples=data_samples,
                                               size=self.size,
                                               size_divisor=self.size_divisor,
                                               pad_val=self.pad_val,
                                               seg_pad_val=self.seg_pad_val)

            if self.batch_augments is not None:
                inputs, data_samples = self.batch_augments(
                    inputs, data_samples)
        else:
            pass
