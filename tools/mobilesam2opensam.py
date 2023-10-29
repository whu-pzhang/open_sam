from collections import OrderedDict
import argparse
from pathlib import Path

import torch


def convert_tinyvit(weight):
    """Weight Converter.

    Converts the weights from timm to mmpretrain
    Args:
        weight (dict): weight dict from timm
    Returns:
        Converted weight dict for mmpretrain
    """
    new_ckpt = OrderedDict()
    mapping = {
        'c.weight': 'conv2d.weight',
        'bn.weight': 'bn2d.weight',
        'bn.bias': 'bn2d.bias',
        'bn.running_mean': 'bn2d.running_mean',
        'bn.running_var': 'bn2d.running_var',
        'bn.num_batches_tracked': 'bn2d.num_batches_tracked',
        'layers': 'stages',
        # 'norm_head': 'norm3',
    }

    for k, v in weight.items():
        # keyword mapping
        for mk, mv in mapping.items():
            if mk in k:
                k = k.replace(mk, mv)

        if k.startswith('head.') or k.startswith('norm_head.'):
            continue
        elif k.startswith('neck.'):
            new_ckpt[k.replace('neck', 'channel_reduction')] = v
        else:
            new_ckpt[k] = v

    return new_ckpt


def convert_weights(ckpt):
    new_ckpt = OrderedDict()

    # extract image cncoder weighs
    image_encoder_ckpt = OrderedDict()
    for k, v in original_model.items():
        if k.startswith('image_encoder'):
            image_encoder_ckpt[k.replace('image_encoder.', '')] = v
        else:
            new_ckpt[k] = v

    new_image_encoder_ckpt = convert_tinyvit(image_encoder_ckpt)
    for k, v in new_image_encoder_ckpt.items():
        new_k = 'image_encoder.' + k
        new_ckpt[new_k] = v

    return new_ckpt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert model keys')
    parser.add_argument('src', help='src detectron model path')
    parser.add_argument('dst', help='save path')
    args = parser.parse_args()
    dst = Path(args.dst)
    if dst.suffix != '.pth':
        print('The path should contain the name of the pth format file.')
        exit(1)
    dst.parent.mkdir(parents=True, exist_ok=True)

    original_model = torch.load(args.src, map_location='cpu')
    # print(original_model.keys())
    converted_model = convert_weights(original_model)
    torch.save(converted_model, args.dst)
