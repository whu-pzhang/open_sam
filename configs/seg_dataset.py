# dataset
dataset_type = 'SegDataset'
data_root = 'data/whu-building/cropped_aerial_data'

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='mmseg.LoadAnnotations', imdecode_backend='pillow'),
    dict(type='RandomResize',
         scale=(2048, 512),
         ratio_range=(0.5, 2.0),
         keep_ratio=True),
    dict(type='mmseg.RandomCrop', crop_size=(512, 512), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='mmseg.PhotoMetricDistortion'),
    dict(type='ResizeLongestEdge', scale=1024),
    dict(type='GenerateSAMPrompt',
         max_instances_per_classes=10,
         points_per_instance=2,
         ignore_values=[0, 255]),
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
