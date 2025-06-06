# dataset
dataset_type = 'mmdet.CocoDataset'
data_root = 'data/nwpu'

metainfo = dict(classes=('airplane', 'storage tank', 'baseball diamond',
                         'tennis court', 'basketball court',
                         'ground track field', 'vehicle', 'harbor', 'bridge',
                         'ship'),
                palette=[(0, 0, 0), (0, 0, 63), (0, 63, 63), (0, 63, 0),
                         (0, 63, 127), (0, 63, 191), (0, 63, 255),
                         (0, 127, 63), (0, 127, 127), (0, 0, 127)])

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='mmdet.Resize', scale=(2048, 512), keep_ratio=True),
    dict(type='mmdet.RandomFlip', prob=0.5),
    dict(type='ResizeLongestEdge', scale=1024),
    dict(type='GenerateSAMPrompt',
         max_instances_per_classes=10,
         points_per_instance=2),
    dict(type='PackSamInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, with_mask=False),
    dict(type='ResizeLongestEdge', scale=1024),
    dict(type='GenerateSAMPrompt',
         prompt_type='boxes',
         noise_cfg=None,
         max_instances_per_classes=10,
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
                 ann_file='annotations/train.json',
                 data_prefix=dict(img='images'),
                 filter_cfg=dict(filter_empty_gt=True, min_size=32),
                 pipeline=train_pipeline))

val_dataloader = dict(batch_size=1,
                      num_workers=4,
                      persistent_workers=True,
                      sampler=dict(type='DefaultSampler', shuffle=False),
                      dataset=dict(type=dataset_type,
                                   data_root=data_root,
                                   metainfo=metainfo,
                                   ann_file='annotations/val.json',
                                   data_prefix=dict(img='images'),
                                   pipeline=test_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(type='IoU')
test_evaluator = val_evaluator
