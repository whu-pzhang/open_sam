# dataset
dataset_type = 'mmdet.CocoDataset'
data_root = 'data/whu-building/cropped_aerial_data'

metainfo = dict(classes=('background', 'building'),
                palette=[[0, 0, 0], [255, 255, 255]])

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='mmdet.Resize', scale=(2048, 512), keep_ratio=True),
    dict(type='mmdet.RandomFlip', prob=0.5),
    dict(type='ResizeLongestEdge', scale=1024),
    dict(
        type='GenerateSAMPrompt',
        prompt_type=['point', 'boxes'],
        # noise_cfg=dict(bbox_std_ratio=0.1, bbox_max_offset=20),
        # prompt_type='boxes',
        noise_cfg=None,
        max_instances_per_classes=10,
        points_per_instance=1),
    dict(type='PackSamInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='ResizeLongestEdge', scale=1024),
    dict(type='GenerateSAMPrompt',
         prompt_type='boxes',
         noise_cfg=None,
         max_instances_per_classes=999,
         points_per_instance=2),
    dict(type='PackSamInputs'),
]

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='mmdet.AspectRatioBatchSampler'),
    dataset=dict(type=dataset_type,
                 data_root=data_root,
                 metainfo=metainfo,
                 ann_file='annotations/whu-building_train.json',
                 data_prefix=dict(img='train/image'),
                 filter_cfg=dict(filter_empty_gt=True, min_size=32),
                 pipeline=train_pipeline))

val_dataloader = dict(batch_size=1,
                      num_workers=4,
                      persistent_workers=True,
                      sampler=dict(type='DefaultSampler', shuffle=False),
                      dataset=dict(
                          type=dataset_type,
                          data_root=data_root,
                          metainfo=metainfo,
                          ann_file='annotations/whu-building_val.json',
                          data_prefix=dict(img='val/image'),
                          filter_cfg=dict(filter_empty_gt=True),
                          test_mode=True,
                          pipeline=test_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(type='IoU')
test_evaluator = val_evaluator
