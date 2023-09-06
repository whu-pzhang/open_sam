from mmengine.structures import BaseDataElement, PixelData, InstanceData


class SamDataSample(BaseDataElement):
    '''
    The attributes in ``SamDataSample`` are divided into several parts:

        - ``gt_masks``(PixelData): Ground truth of instance masks.
        - ``pred_masks``(PixelData): Prediction of instance masks.
        - ``pred_logits``(PixelData): Predicted logits of instance masks.
    '''

    @property
    def gt_masks(self) -> PixelData:
        return self._gt_masks

    @gt_masks.setter
    def gt_masks(self, value: PixelData):
        self.set_field(value, '_gt_masks', dtype=PixelData)

    @gt_masks.deleter
    def gt_masks(self):
        del self._gt_masks

    @property
    def pred_masks(self) -> PixelData:
        return self._pred_masks

    @pred_masks.setter
    def pred_masks(self, value: PixelData):
        self.set_field(value, '_pred_masks', dtype=PixelData)

    @gt_masks.deleter
    def pred_masks(self):
        del self._pred_masks

    @property
    def pred_logits(self) -> PixelData:
        return self._pred_logits

    @pred_logits.setter
    def seg_logits(self, value: PixelData) -> None:
        self.set_field(value, '_pred_logits', dtype=PixelData)

    @seg_logits.deleter
    def seg_logits(self) -> None:
        del self._pred_logits

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
