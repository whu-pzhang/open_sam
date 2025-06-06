from mmengine.runner.checkpoint import load_checkpoint

from open_sam.registry import MODELS
from open_sam.utils import register_all_modules

register_all_modules()

# yapf: disable
model_zoo = {
    'edge':'weights/edge-sam-3x-5a3e3261.pth',
    'tiny': 'weights/sam_vit-tiny-6dbb9052.pth', # noqa
    'base': 'https://download.openmmlab.com/mmsegmentation/v0.5/sam/sam_vit-base-p16_3rdparty_sa1b-1024x1024_20230413-78a25eed.pth',  # noqa
    'large': 'https://download.openmmlab.com/mmsegmentation/v0.5/sam/sam_vit-large-p16_3rdparty_sa1b-1024x1024_20230413-940520da.pth',  # noqa
    'huge': 'https://download.openmmlab.com/mmsegmentation/v0.5/sam/sam_vit-huge-p16_3rdparty_sa1b-1024x1024_20230413-faaf96f6.pth',  # noqa
}
# yapf: enable


def build_sam_vit_h(checkpoint=None):
    return build_sam(arch='huge', checkpoint=checkpoint)


def build_sam_vit_l(checkpoint=None):
    return build_sam(arch='large', checkpoint=checkpoint)


def build_sam_vit_b(checkpoint=None):
    return build_sam(arch='base', checkpoint=checkpoint)


def build_sam_vit_t(checkpoint=None):
    return build_sam(arch='tiny', checkpoint=checkpoint)


def build_edge_sam(checkpoint=None):
    return build_sam(arch='edge', checkpoint=checkpoint)


def build_sam(arch='huge', checkpoint=None):
    if arch == 'edge':
        image_encoder = dict(type='RepViT',
                             arch='m1',
                             img_size=1024,
                             out_channels=256)
    elif arch == 'tiny':
        image_encoder = dict(type='TinyViT',
                             arch='5m',
                             img_size=1024,
                             out_channels=256,
                             mlp_ratio=4.,
                             drop_rate=0.,
                             drop_path_rate=0.,
                             use_checkpoint=False,
                             mbconv_expand_ratio=4.0,
                             local_conv_size=3,
                             layer_lr_decay=0.8)
    else:
        image_encoder = dict(type='ViTSAM',
                             arch=arch,
                             img_size=1024,
                             patch_size=16,
                             out_channels=256,
                             use_abs_pos=True,
                             use_rel_pos=True,
                             window_size=14)
    model = MODELS.build(
        dict(type='SAM',
             image_encoder=image_encoder,
             prompt_encoder=dict(type='PromptEncoder',
                                 embed_dim=256,
                                 image_embedding_size=(64, 64),
                                 input_image_size=(1024, 1024),
                                 mask_in_chans=16),
             mask_decoder=dict(type='MaskDecoder',
                               num_multimask_outputs=3,
                               transformer=dict(type='TwoWayTransformer',
                                                depth=2,
                                                embedding_dim=256,
                                                mlp_dim=2048,
                                                num_heads=8),
                               transformer_dim=256,
                               iou_head_depth=3,
                               iou_head_hidden_dim=256)))
    model.eval()
    if checkpoint is not None:
        model_url = checkpoint
    else:
        model_url = model_zoo.get(arch)

    load_checkpoint(model, model_url, strict=True)
    return model


sam_model_registry = {
    "default": build_sam_vit_h,
    "vit_h": build_sam_vit_h,
    "vit_l": build_sam_vit_l,
    "vit_b": build_sam_vit_b,
    "mobile_sam": build_sam_vit_t,
    "edge_sam": build_edge_sam,
}
