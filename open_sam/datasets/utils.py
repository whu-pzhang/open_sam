from torch.utils.data.dataloader import default_collate
from mmengine.dataset import COLLATE_FUNCTIONS


@COLLATE_FUNCTIONS.register_module()
def custom_collate_fn(batch):
    elem = batch[0]
    keep_keys = ['gt_masks', 'boxes', 'point_coords', 'img_shape', 'ori_shape']
    ret_batch = {
        key: default_collate([d[key] for d in batch])
        for key in elem if key not in keep_keys
    }
    box_points = {
        key: [default_collate(d[key]) for d in batch]
        for key in keep_keys
    }

    ret_batch.update(box_points)

    return ret_batch


@COLLATE_FUNCTIONS.register_module()
def sam_collate_fn(batch):
    elem = batch[0]  # dict: inputs, data_samples

    keep_keys = ['gt_masks', 'boxes', 'point_coords', 'img_shape', 'ori_shape']
    ret_batch = {
        key: default_collate([d[key] for d in batch])
        for key in elem if key not in keep_keys
    }
    box_points = {
        key: [default_collate(d[key]) for d in batch]
        for key in keep_keys
    }

    ret_batch.update(box_points)

    return ret_batch
