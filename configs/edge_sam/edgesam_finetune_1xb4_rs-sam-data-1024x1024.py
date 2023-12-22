_base_ = [
    '../_base_/datasets/rs_sam_data.py',
    '../_base_/default_runtime.py',
    '../_base_/models/edge_sam.py',
]

# training schedule
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=12, val_interval=2)
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
optim_wrapper = dict(type='OptimWrapper',
                     optimizer=dict(type='Adam',
                                    lr=1e-4,
                                    betas=(0.9, 0.999),
                                    eps=1e-08,
                                    weight_decay=0))

default_hooks = dict(checkpoint=dict(interval=1, max_keep_ckpts=2))

randomness = dict(seed=3407)
