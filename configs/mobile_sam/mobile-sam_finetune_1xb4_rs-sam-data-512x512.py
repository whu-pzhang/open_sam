_base_ = [
    '../_base_/datasets/rs_sam_data.py',
    '../_base_/default_runtime.py',
    '../_base_/models/mobile_sam.py',
]

data_preprocessor = dict(type='SamDataPreprocessor',
                         mean=[123.675, 116.28, 103.53],
                         std=[58.395, 57.12, 57.375],
                         bgr_to_rgb=True,
                         size=(512, 512))
model = dict(type='SAM',
             data_preprocessor=data_preprocessor,
             image_encoder=dict(img_size=512),
             prompt_encoder=dict(type='PromptEncoder',
                                 image_embedding_size=(32, 32),
                                 input_image_size=(512, 512)))

# training schedule
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=12, val_interval=13)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# learning rate
param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0,
         end=500),
    dict(type='MultiStepLR',
         begin=0,
         end=12,
         by_epoch=True,
         milestones=[7, 10],
         gamma=0.5)
]

# optimizer
optim_wrapper = dict(
    type='AmpOptimWrapper',
    dtype='float16',  # (float16, bfloat16, None)
    optimizer=dict(type='Adam',
                   lr=1e-4,
                   betas=(0.9, 0.999),
                   eps=1e-08,
                   weight_decay=0))

default_hooks = dict(checkpoint=dict(interval=1, max_keep_ckpts=2))

randomness = dict(seed=3407)
