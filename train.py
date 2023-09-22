# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import logging
import os
import os.path as osp

from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmengine.runner import Runner

from mmseg.registry import RUNNERS, MODELS

import open_sam

from mmseg.utils import register_all_modules


def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume',
        action='store_true',
        default=False,
        help='resume from the latest checkpoint in the work_dir automatically')
    parser.add_argument('--amp',
                        action='store_true',
                        default=False,
                        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument('--launcher',
                        choices=['none', 'pytorch', 'slurm', 'mpi'],
                        default='none',
                        help='job launcher')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()

    print(MODELS)

    # load config
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    # enable automatic-mixed-precision training
    if args.amp is True:
        optim_wrapper = cfg.optim_wrapper.type
        if optim_wrapper == 'AmpOptimWrapper':
            print_log('AMP training is already enabled in your config.',
                      logger='current',
                      level=logging.WARNING)
        else:
            assert optim_wrapper == 'OptimWrapper', (
                '`--amp` is only supported when the optimizer wrapper type is '
                f'`OptimWrapper` but got {optim_wrapper}.')
            cfg.optim_wrapper.type = 'AmpOptimWrapper'
            cfg.optim_wrapper.loss_scale = 'dynamic'

    # resume training
    cfg.resume = args.resume

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    # start training
    runner.train()


def main2():
    from mmseg.registry import MODELS, DATASETS
    from torch.optim import AdamW
    from mmengine.optim import AmpOptimWrapper

    # model
    def build_sam(arch):
        import torch
        from mmengine.runner.checkpoint import load_checkpoint
        from open_sam.sam_predictor import model_zoo

        cfg = dict(
            type='SAM',
            image_encoder=dict(type='mmpretrain.ViTSAM',
                               arch=arch,
                               img_size=1024,
                               patch_size=16,
                               out_channels=256,
                               use_abs_pos=True,
                               use_rel_pos=True,
                               window_size=14),
            prompt_encoder=dict(
                type='PromptEncoder',
                embed_dim=256,
                image_embedding_size=(64, 64),
                input_image_size=(1024, 1024),
                mask_in_chans=16,
            ),
            mask_decoder=dict(
                type='MaskDecoder',
                num_multimask_outputs=3,
                transformer=dict(
                    type='TwoWayTransformer',
                    depth=2,
                    embedding_dim=256,
                    mlp_dim=2048,
                    num_heads=8,
                ),
                transformer_dim=256,
                iou_head_depth=3,
                iou_head_hidden_dim=256,
            ),
            loss_decode=[
                dict(type='CrossEntropyLoss',
                     use_sigmoid=True,
                     avg_non_ignore=True),
                dict(type='DiceLoss'),
            ],
        )

        model = MODELS.build(cfg)

        load_checkpoint(model, model_zoo.get(arch), strict=True)
        if torch.cuda.is_available():
            model = model.cuda()
        return model

    model = build_sam(arch='base')
    for p in model.image_encoder.parameters():
        p.requires_grad = False
    for p in model.prompt_encoder.parameters():
        p.requires_grad = False

    # dataset
    from torchvision.transforms import Compose

    from open_sam.datasets.transforms import ResizeLongestSide, PackInputs

    transform = Compose([
        ResizeLongestSide(1024),
        PackInputs(),
    ])
    dataset_cfg = dict(type='WHUBuildingDataset',
                       data_root='data/whu-building/cropped_aerial_data',
                       split='train',
                       ann_file='data/whu-building/train.txt',
                       max_objects=15,
                       points_per_instance=1,
                       transform=transform)
    ds = DATASETS.build(dataset_cfg)

    train_dataloader = dict(dataset=ds,
                            batch_size=2,
                            num_workers=0,
                            sampler=dict(type='DefaultSampler', shuffle=True),
                            drop_last=True,
                            collate_fn=dict(type='custom_collate_fn'))

    runner = Runner(
        model=model,
        work_dir='./work_dir',
        train_dataloader=train_dataloader,
        optim_wrapper=dict(type='AmpOptimWrapper',
                           optimizer=dict(type='Adam', lr=1e-5)),
        train_cfg=dict(by_epoch=True, max_epochs=10),
        default_hooks=dict(timer=dict(type='IterTimerHook'),
                           logger=dict(type='LoggerHook',
                                       interval=50,
                                       log_metric_by_epoch=False),
                           checkpoint=dict(type='CheckpointHook', interval=1)),
    )

    runner.train()


if __name__ == '__main__':
    main()
    # main2()
