from typing import List, Optional

from mmengine.structures import BaseDataElement, PixelData, InstanceData


class SamDataSample(BaseDataElement):
    '''
    The attributes in ``SamDataSample`` are divided into several parts:

        - ``gt_instances``(InstanceData): Ground truth of instance annotations.
        - ``pred_instances``(InstanceData): Instances of detection predictions.

        - ``prompt``(PixelData): Ground truth of instance masks.
        - ``pred_masks``(PixelData): Prediction of instance masks.
        - ``pred_logits``(PixelData): Predicted logits of instance masks.

        - ``gt_sem_seg``(PixelData): Ground truth of semantic segmentation.
        - ``pred_sem_seg``(PixelData): Prediction of semantic segmentation.
    '''

    @property
    def prompt_instances(self) -> InstanceData:
        return self._prompt_instances

    @prompt_instances.setter
    def prompt_instances(self, value: InstanceData):
        self.set_field(value, '_prompt_instances', dtype=InstanceData)

    @prompt_instances.deleter
    def prompt_instances(self):
        del self._prompt_instances

    @property
    def gt_instances(self) -> InstanceData:
        return self._gt_instances

    @gt_instances.setter
    def gt_instances(self, value: InstanceData):
        self.set_field(value, '_gt_instances', dtype=InstanceData)

    @gt_instances.deleter
    def gt_instances(self):
        del self._gt_instances

    @property
    def pred_instances(self) -> InstanceData:
        return self._pred_instances

    @pred_instances.setter
    def pred_instances(self, value: InstanceData):
        self.set_field(value, '_pred_instances', dtype=InstanceData)

    @pred_instances.deleter
    def pred_instances(self):
        del self._pred_instances

    @property
    def gt_sem_seg(self) -> PixelData:
        return self._gt_sem_seg

    @gt_sem_seg.setter
    def gt_sem_seg(self, value: PixelData):
        self.set_field(value, '_gt_sem_seg', dtype=PixelData)

    @gt_sem_seg.deleter
    def gt_sem_seg(self):
        del self._gt_sem_seg

    @property
    def pred_sem_seg(self) -> PixelData:
        return self._pred_sem_seg

    @pred_sem_seg.setter
    def pred_sem_seg(self, value: PixelData):
        self.set_field(value, '_pred_sem_seg', dtype=PixelData)

    @pred_sem_seg.deleter
    def pred_sem_seg(self):
        del self._pred_sem_seg


SampleList = List[SamDataSample]
OptSampleList = Optional[SampleList]
