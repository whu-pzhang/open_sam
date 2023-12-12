import os
from collections import OrderedDict
from hashlib import sha256
import argparse
from pathlib import Path

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


def convert_image_encoder(ckpt):
    new_ckpt = OrderedDict()

    for k, v in ckpt.items():
        if k.startswith('patch_embed'):
            if 'proj' in k:
                new_k = k.replace('proj', 'projection')
            else:
                new_k = k

        elif k.startswith('blocks'):
            if 'norm' in k:
                new_k = k.replace('norm', 'ln')
            elif 'mlp.lin1' in k:
                new_k = k.replace('mlp.lin1', 'ffn.layers.0.0')
            elif 'mlp.lin2' in k:
                new_k = k.replace('mlp.lin2', 'ffn.layers.1')
            else:
                new_k = k
            new_k = new_k.replace('blocks.', 'layers.')
        elif k.startswith('neck'):
            new_k = k.replace('neck', 'channel_reduction')
        else:
            new_k = k
        new_ckpt[new_k] = v

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

    new_image_encoder_ckpt = convert_image_encoder(image_encoder_ckpt)
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
    parser.add_argument('src', help='src detectron model path')
    parser.add_argument('dst', help='save path')
    args = parser.parse_args()
    dst = Path(args.dst)
    if dst.suffix != '.pth':
        print('The path should contain the name of the pth format file.')
        exit(1)
    dst.parent.mkdir(parents=True, exist_ok=True)

    process_checkpoint(args.src, args.dst)
