from collections import OrderedDict
import argparse
from pathlib import Path
import os
from hashlib import sha256

import torch

BLOCK_SIZE = 128 * 1024


def sha256sum(filename: str) -> str:
    """Compute SHA256 message digest from a file."""
    hash_func = sha256()
    byte_array = bytearray(BLOCK_SIZE)
    memory_view = memoryview(byte_array)
    with open(filename, 'rb', buffering=0) as file:
        for block in iter(lambda: file.readinto(memory_view), 0):
            hash_func.update(memory_view[:block])
    return hash_func.hexdigest()


def convert_repvit(weight):
    """Weight Converter.

    Converts the weights from timm to mmpretrain
    Args:
        weight (dict): weight dict from timm
    Returns:
        Converted weight dict for mmpretrain
    """
    new_ckpt = OrderedDict()
    mapping = {
        'fuse_stage3.op_list.0.weight': 'fuse_stage3.0.weight',
    }

    for k, v in weight.items():
        # keyword mapping
        for mk, mv in mapping.items():
            if mk in k:
                k = k.replace(mk, mv)
        new_ckpt[k] = v

    return new_ckpt


def convert_weights(ckpt):
    new_ckpt = OrderedDict()

    # extract image cncoder weighs
    image_encoder_ckpt = OrderedDict()
    for k, v in ckpt.items():
        if k.startswith('image_encoder'):
            image_encoder_ckpt[k.replace('image_encoder.', '')] = v
        else:
            new_ckpt[k] = v

    new_image_encoder_ckpt = convert_repvit(image_encoder_ckpt)
    for k, v in new_image_encoder_ckpt.items():
        new_k = 'image_encoder.' + k
        new_ckpt[new_k] = v

    return new_ckpt


def process_checkpoint(in_file, out_file):
    original_model = torch.load(in_file, map_location='cpu')
    converted_model = convert_weights(original_model)
    torch.save(converted_model, out_file)
    sha = sha256sum(in_file)
    final_file = out_file.rstrip('.pth') + f'-{sha[:8]}.pth'
    os.rename(out_file, final_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert model keys')
    parser.add_argument('src', help='src model path')
    parser.add_argument('dst', help='save path')
    args = parser.parse_args()
    dst = Path(args.dst)
    if dst.suffix != '.pth':
        print('The path should contain the name of the pth format file.')
        exit(1)
    dst.parent.mkdir(parents=True, exist_ok=True)

    process_checkpoint(args.src, args.dst)
