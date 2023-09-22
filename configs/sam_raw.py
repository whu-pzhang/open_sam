# dataset
dataset_type = 'SegDataset'
data_root = 'data/whu-building/cropped_aerial_data'

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', imdecode_backend='pillow'),
    dict(type='RandomResize',
         scale=(2048, 512),
         ratio_range=(0.5, 2.0),
         keep_ratio=True),
    dict(type='RandomCrop', crop_size=(512, 512), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='ResizeLongestEdge', scale=1024),
    dict(type='GenerateSAMPrompt', max_instances=15, points_per_instance=2),
    # dict(type='ResizeLongestSide', target_length=1024),
    dict(type='PackSamInputs'),
]

test_pipeline = []

train_dataloader = dict(batch_size=2,
                        num_workers=4,
                        persistent_workers=True,
                        sampler=dict(type='InfiniteSampler', shuffle=True),
                        dataset=dict(type=dataset_type,
                                     data_root=data_root,
                                     img_suffix='.tif',
                                     seg_map_suffix='.tif',
                                     data_prefix=dict(
                                         img_path='train/image',
                                         seg_map_path='train/label'),
                                     ann_file='../train.txt',
                                     pipeline=train_pipeline))

val_dataloader = dict(batch_size=1,
                      num_workers=4,
                      persistent_workers=True,
                      sampler=dict(type='DefaultSampler', shuffle=False),
                      dataset=dict(type=dataset_type,
                                   data_root=data_root,
                                   img_suffix='.tif',
                                   seg_map_suffix='.tif',
                                   data_prefix=dict(img_path='val/image',
                                                    seg_map_path='val/label'),
                                   ann_file='../val.txt',
                                   pipeline=test_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(type='IoU')
test_evaluator = val_evaluator

# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer, clip_grad=None)

# learning policy
param_scheduler = [
    dict(type='PolyLR',
         eta_min=1e-4,
         power=0.9,
         begin=0,
         end=20000,
         by_epoch=False)
]

# training schedule
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=12, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1),
)

# runtime
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

log_processor = dict(by_epoch=False)
log_level = 'INFO'
load_from = None
resume = False
