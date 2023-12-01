from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
import numpy as np
from mmengine.utils import is_seq_of
from mmengine.model import BaseDataPreprocessor, ImgDataPreprocessor
from mmdet.models.utils.misc import samplelist_boxtype2tensor
from mmdet.models.data_preprocessors import DetDataPreprocessor

from open_sam.registry import MODELS
from ..datasets import SamDataSample


@MODELS.register_module()
class SamDataPreprocessor(ImgDataPreprocessor):
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
                 mean=None,
                 std=None,
                 pad_value: int = 0,
                 bgr_to_rgb: bool = False,
                 rgb_to_bgr: bool = False,
                 pad_size: Tuple[int] = (1024, 1024),
                 point_pad_value: int = -10,
                 non_blocking: Optional[bool] = False):
        super().__init__(mean=mean,
                         std=std,
                         bgr_to_rgb=bgr_to_rgb,
                         rgb_to_bgr=rgb_to_bgr,
                         non_blocking=non_blocking)
        self.pad_size = pad_size
        self.pad_value = pad_value
        self.point_pad_value = point_pad_value

    def forward(self, data: dict, training: bool = False):
        """Perform normalization,padding and bgr2rgb conversion based on
        ``BaseDataPreprocessor``.

        Args:
            data (dict): Data sampled from dataloader.
            training (bool): Whether to enable training time augmentation.

        Returns:
            dict: Data in the same format as the model input.
        """
        # batch_pad_shape = self._get_pad_shape(data)

        data = super().forward(data=data, training=training)

        inputs, data_samples = data['inputs'], data['data_samples']
        inputs = self.pad_images(inputs)

        # self.pad_points_and_labels(data_samples)

        return {'inputs': inputs, 'data_samples': data_samples}

    def _get_pad_shape(self, data: dict) -> List[tuple]:
        """Get the pad_shape of each image based on data and
        pad_size_divisor."""
        _batch_inputs = data['inputs']
        # Process data with `pseudo_collate`.
        if is_seq_of(_batch_inputs, torch.Tensor):
            batch_pad_shape = []
            for ori_input in _batch_inputs:
                pad_h = self.pad_size[1] - ori_input.shape[1]
                pad_w = self.pad_size[0] - ori_input.shape[2]
                batch_pad_shape.append((pad_h, pad_w))
        # Process data with `default_collate`.
        elif isinstance(_batch_inputs, torch.Tensor):
            assert _batch_inputs.dim() == 4, (
                'The input of `ImgDataPreprocessor` should be a NCHW tensor '
                'or a list of tensor, but got a tensor with shape: '
                f'{_batch_inputs.shape}')
            pad_h = self.pad_size[1] - _batch_inputs.shape[2]
            pad_w = self.pad_size[0] - _batch_inputs.shape[3]
            batch_pad_shape = [(pad_h, pad_w)] * _batch_inputs.shape[0]
        else:
            raise TypeError('Output of `cast_data` should be a dict '
                            'or a tuple with inputs and data_samples, but got'
                            f'{type(data)}: {data}')
        return batch_pad_shape

    def pad_images(self, batch_inputs):
        h, w = batch_inputs.shape[2:]
        target_w, target_h = self.pad_size
        pad_h = target_h - h
        pad_w = target_w - w
        batch_inputs = F.pad(batch_inputs, (0, pad_w, 0, pad_h), 'constant',
                             self.pad_value)
        return batch_inputs

    def pad_points_and_labels(self, data_samples):
        '''
        input_points (`torch.FloatTensor` of shape `(batch_size, num_points, 2)`):
            Input 2D spatial points, this is used by the prompt encoder to encode the prompt. Generally yields to much
            better results. The points can be obtained by passing a list of list of list to the processor that will
            create corresponding `torch` tensors of dimension 4. The first dimension is the image batch size, the
            second dimension is the point batch size (i.e. how many segmentation masks do we want the model to predict
            per input point), the third dimension is the number of points per segmentation mask (it is possible to pass
            multiple points for a single mask), and the last dimension is the x (vertical) and y (horizontal)
            coordinates of the point. If a different number of points is passed either for each image, or for each
            mask, the processor will create "PAD" points that will correspond to the (0, 0) coordinate, and the
            computation of the embedding will be skipped for these points using the labels.
        input_labels (`torch.LongTensor` of shape `(batch_size, point_batch_size, num_points)`):
            Input labels for the points, this is used by the prompt encoder to encode the prompt. According to the
            official implementation, there are 3 types of labels

            - `1`: the point is a point that contains the object of interest
            - `0`: the point is a point that does not contain the object of interest
            - `-1`: the point corresponds to the background

            We added the label:

            - `-10`: the point is a padding point, thus should be ignored by the prompt encoder

            The padding labels should be automatically done by the processor.
        '''
        point_coords = [
            data_sample.prompt_instances.point_coords
            for data_sample in data_samples
        ]
        point_labels = [
            data_sample.prompt_instances.point_labels
            for data_sample in data_samples
        ]

        # expected number of points in batch
        expected_nb_points = max(pc.shape[0] for pc in point_coords)
        processed_input_points = []
        device = point_coords[0].device
        for i, pc in enumerate(point_coords):
            if pc.shape[0] != expected_nb_points:
                pad_len = expected_nb_points - pc.shape[0]
                pad_pc = torch.full(size=(pad_len, pc.shape[1], 2),
                                    fill_value=self.point_pad_value,
                                    device=device)
                pc = torch.concat([pc, pad_pc + self.point_pad_value], dim=0)

                pad_labels = torch.full(size=(pad_len, 2),
                                        fill_value=self.point_pad_value,
                                        device=device)
                pl = torch.concat([point_labels[i], pad_labels], dim=0)

            processed_input_points.append(pc)

    def pad_boxes(self, data_samples):
        pass

    def pad_gt_masks(self,
                     batch_data_samples: Sequence[SamDataSample]) -> None:
        """Pad gt_masks to shape of batch_input_shape."""
        if 'masks' in batch_data_samples[0].gt_instances:
            for data_samples in batch_data_samples:
                masks = data_samples.gt_instances.masks
                data_samples.gt_instances.masks = masks.pad(
                    data_samples.batch_input_shape,
                    pad_val=self.mask_pad_value)
