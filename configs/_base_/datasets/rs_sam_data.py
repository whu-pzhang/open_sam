# dataset
dataset_type = 'SamDataset'
data_root = 'data/rs_sam_data'

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='mmdet.Resize', scale=(2048, 1024), keep_ratio=True),
    dict(type='mmdet.RandomFlip', prob=0.5),
    dict(type='ResizeLongestEdge', scale=1024),
    dict(
        type='GenerateSAMPrompt',
        prompt_type=['point', 'bbox'],
        # noise_cfg=dict(bbox_std_ratio=0.1, bbox_max_offset=20),
        max_instances_per_classes=10,
        points_per_instance=1,
        noise_cfg=None),
    dict(type='PackSamInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    # dict(type='ResizeLongestEdge', scale=1024),
    dict(type='GenerateSAMPrompt',
         prompt_type='bbox',
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
                 data_prefix=dict(img_path='train_whu-building',
                                  json_path='train_whu-building'),
                 filter_cfg=dict(min_size=32),
                 pipeline=train_pipeline))

val_dataloader = dict(batch_size=1,
                      num_workers=4,
                      persistent_workers=True,
                      sampler=dict(type='DefaultSampler', shuffle=False),
                      dataset=dict(type=dataset_type,
                                   data_root=data_root,
                                   data_prefix=dict(
                                       img_path='val_whu-building',
                                       json_path='val_whu-building'),
                                   test_mode=True,
                                   pipeline=test_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(type='ClassAgnosticIoU', iou_metrics=['mIoU', 'mDice'])
test_evaluator = val_evaluator
