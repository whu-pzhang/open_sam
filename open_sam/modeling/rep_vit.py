from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import SqueezeExcite
from mmengine.model.weight_init import trunc_normal_
from mmengine.model import BaseModule
from mmpretrain.models.utils import to_2tuple

from open_sam.registry import MODELS
from .common import LayerNorm2d
from .utils import make_divisible


class Conv2d_BN(nn.Sequential):

    def __init__(self,
                 a,
                 b,
                 ks=1,
                 stride=1,
                 pad=0,
                 dilation=1,
                 groups=1,
                 bn_weight_init=1):
        super().__init__()
        self.add_module(
            'c',
            torch.nn.Conv2d(a,
                            b,
                            ks,
                            stride,
                            pad,
                            dilation,
                            groups,
                            bias=False))
        self.add_module('bn', torch.nn.BatchNorm2d(b))
        nn.init.constant_(self.bn.weight, bn_weight_init)
        nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps) ** 0.5
        m = nn.Conv2d(w.size(1) * self.c.groups,
                      w.size(0),
                      w.shape[2:],
                      stride=self.c.stride,
                      padding=self.c.padding,
                      dilation=self.c.dilation,
                      groups=self.c.groups,
                      device=c.weight.device)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class Residual(torch.nn.Module):

    def __init__(self, m, drop=0.):
        super().__init__()
        self.m = m
        self.drop = drop

    def forward(self, x):
        if self.training and self.drop > 0:
            return x + self.m(x) * torch.rand(
                x.size(0), 1, 1, 1, device=x.device).ge_(
                    self.drop).div(1 - self.drop).detach()
        else:
            return x + self.m(x)

    @torch.no_grad()
    def fuse(self):
        if isinstance(self.m, Conv2d_BN):
            m = self.m.fuse()
            assert (m.groups == m.in_channels)
            identity = torch.ones(m.weight.shape[0], m.weight.shape[1], 1, 1)
            identity = F.pad(identity, [1, 1, 1, 1])
            m.weight += identity.to(m.weight.device)
            return m
        elif isinstance(self.m, nn.Conv2d):
            m = self.m
            assert (m.groups != m.in_channels)
            identity = torch.ones(m.weight.shape[0], m.weight.shape[1], 1, 1)
            identity = F.pad(identity, [1, 1, 1, 1])
            m.weight += identity.to(m.weight.device)
            return m
        else:
            return self


class RepVGGDW(torch.nn.Module):

    def __init__(self, ed) -> None:
        super().__init__()
        self.conv = Conv2d_BN(ed, ed, 3, 1, 1, groups=ed)
        self.conv1 = Conv2d_BN(ed, ed, 1, 1, 0, groups=ed)
        self.dim = ed

    def forward(self, x):
        return self.conv(x) + self.conv1(x) + x

    @torch.no_grad()
    def fuse(self):
        conv = self.conv.fuse()
        conv1 = self.conv1.fuse()

        conv_w = conv.weight
        conv_b = conv.bias
        conv1_w = conv1.weight
        conv1_b = conv1.bias

        conv1_w = F.pad(conv1_w, [1, 1, 1, 1])

        identity = F.pad(
            torch.ones(conv1_w.shape[0],
                       conv1_w.shape[1],
                       1,
                       1,
                       device=conv1_w.device), [1, 1, 1, 1])

        final_conv_w = conv_w + conv1_w + identity
        final_conv_b = conv_b + conv1_b

        conv.weight.data.copy_(final_conv_w)
        conv.bias.data.copy_(final_conv_b)
        return conv


class RepViTBlock(nn.Module):

    def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_se,
                 use_hs):
        super(RepViTBlock, self).__init__()
        assert stride in [1, 2]

        self.identity = stride == 1 and inp == oup
        assert (hidden_dim == 2 * inp)

        if stride == 2:
            self.token_mixer = nn.Sequential(
                Conv2d_BN(inp,
                          inp,
                          kernel_size,
                          stride, (kernel_size - 1) // 2,
                          groups=inp),
                SqueezeExcite(inp, 0.25) if use_se else nn.Identity(),
                Conv2d_BN(inp, oup, ks=1, stride=1, pad=0))
            self.channel_mixer = Residual(
                nn.Sequential(
                    # pw
                    Conv2d_BN(oup, 2 * oup, 1, 1, 0),
                    nn.GELU() if use_hs else nn.GELU(),
                    # pw-linear
                    Conv2d_BN(2 * oup, oup, 1, 1, 0, bn_weight_init=0),
                ))
        else:
            assert (self.identity)
            self.token_mixer = nn.Sequential(
                RepVGGDW(inp),
                SqueezeExcite(inp, 0.25) if use_se else nn.Identity(),
            )
            self.channel_mixer = Residual(
                nn.Sequential(
                    # pw
                    Conv2d_BN(inp, hidden_dim, 1, 1, 0),
                    nn.GELU() if use_hs else nn.GELU(),
                    # pw-linear
                    Conv2d_BN(hidden_dim, oup, 1, 1, 0, bn_weight_init=0),
                ))

    def forward(self, x):
        return self.channel_mixer(self.token_mixer(x))


class BN_Linear(torch.nn.Sequential):

    def __init__(self, a, b, bias=True, std=0.02):
        super().__init__()
        self.add_module('bn', torch.nn.BatchNorm1d(a))
        self.add_module('l', torch.nn.Linear(a, b, bias=bias))
        trunc_normal_(self.l.weight, std=std)
        if bias:
            torch.nn.init.constant_(self.l.bias, 0)

    @torch.no_grad()
    def fuse(self):
        bn, l = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        b = bn.bias - self.bn.running_mean * \
            self.bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = l.weight * w[None, :]
        if l.bias is None:
            b = b @ self.l.weight.T
        else:
            b = (l.weight @ b[:, None]).view(-1) + self.l.bias
        m = torch.nn.Linear(w.size(1), w.size(0), device=l.weight.device)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


@MODELS.register_module()
class RepViT(BaseModule):
    #  [kernel size, t, out_channels, with_se, with_hs, stride]
    arch_settings = {
        'm1': [
            [3, 2, 48, 1, 0, 1],
            [3, 2, 48, 0, 0, 1],
            [3, 2, 48, 0, 0, 1],
            [3, 2, 96, 0, 0, 2],
            [3, 2, 96, 1, 0, 1],
            [3, 2, 96, 0, 0, 1],
            [3, 2, 96, 0, 0, 1],
            [3, 2, 192, 0, 1, 2],
            [3, 2, 192, 1, 1, 1],
            [3, 2, 192, 0, 1, 1],
            [3, 2, 192, 1, 1, 1],
            [3, 2, 192, 0, 1, 1],
            [3, 2, 192, 1, 1, 1],
            [3, 2, 192, 0, 1, 1],
            [3, 2, 192, 1, 1, 1],
            [3, 2, 192, 0, 1, 1],
            [3, 2, 192, 1, 1, 1],
            [3, 2, 192, 0, 1, 1],
            [3, 2, 192, 1, 1, 1],
            [3, 2, 192, 0, 1, 1],
            [3, 2, 192, 1, 1, 1],
            [3, 2, 192, 0, 1, 1],
            [3, 2, 192, 0, 1, 1],
            [3, 2, 384, 0, 1, 2],
            [3, 2, 384, 1, 1, 1],
            [3, 2, 384, 0, 1, 1],
        ],
        'm2': [
            # k, t, c, SE, HS, s
            [3, 2, 64, 1, 0, 1],
            [3, 2, 64, 0, 0, 1],
            [3, 2, 64, 0, 0, 1],
            [3, 2, 128, 0, 0, 2],
            [3, 2, 128, 1, 0, 1],
            [3, 2, 128, 0, 0, 1],
            [3, 2, 128, 0, 0, 1],
            [3, 2, 256, 0, 1, 2],
            [3, 2, 256, 1, 1, 1],
            [3, 2, 256, 0, 1, 1],
            [3, 2, 256, 1, 1, 1],
            [3, 2, 256, 0, 1, 1],
            [3, 2, 256, 1, 1, 1],
            [3, 2, 256, 0, 1, 1],
            [3, 2, 256, 1, 1, 1],
            [3, 2, 256, 0, 1, 1],
            [3, 2, 256, 1, 1, 1],
            [3, 2, 256, 0, 1, 1],
            [3, 2, 256, 1, 1, 1],
            [3, 2, 256, 0, 1, 1],
            [3, 2, 256, 0, 1, 1],
            [3, 2, 512, 0, 1, 2],
            [3, 2, 512, 1, 1, 1],
            [3, 2, 512, 0, 1, 1]
        ],
        'm3': [
            # k, t, c, SE, HS, s
            [3, 2, 64, 1, 0, 1],
            [3, 2, 64, 0, 0, 1],
            [3, 2, 64, 1, 0, 1],
            [3, 2, 64, 0, 0, 1],
            [3, 2, 64, 0, 0, 1],
            [3, 2, 128, 0, 0, 2],
            [3, 2, 128, 1, 0, 1],
            [3, 2, 128, 0, 0, 1],
            [3, 2, 128, 1, 0, 1],
            [3, 2, 128, 0, 0, 1],
            [3, 2, 128, 0, 0, 1],
            [3, 2, 256, 0, 1, 2],
            [3, 2, 256, 1, 1, 1],
            [3, 2, 256, 0, 1, 1],
            [3, 2, 256, 1, 1, 1],
            [3, 2, 256, 0, 1, 1],
            [3, 2, 256, 1, 1, 1],
            [3, 2, 256, 0, 1, 1],
            [3, 2, 256, 1, 1, 1],
            [3, 2, 256, 0, 1, 1],
            [3, 2, 256, 1, 1, 1],
            [3, 2, 256, 0, 1, 1],
            [3, 2, 256, 1, 1, 1],
            [3, 2, 256, 0, 1, 1],
            [3, 2, 256, 1, 1, 1],
            [3, 2, 256, 0, 1, 1],
            [3, 2, 256, 1, 1, 1],
            [3, 2, 256, 0, 1, 1],
            [3, 2, 256, 1, 1, 1],
            [3, 2, 256, 0, 1, 1],
            [3, 2, 256, 0, 1, 1],
            [3, 2, 512, 0, 1, 2],
            [3, 2, 512, 1, 1, 1],
            [3, 2, 512, 0, 1, 1]
        ]
    }

    def __init__(self,
                 arch: str = 'm1',
                 img_size: int = 1024,
                 in_channels: int = 3,
                 out_channels: int = 256,
                 out_indices: int = -1,
                 frozen_stages: int = -1,
                 interpolate_mode: str = 'bicubic',
                 init_cfg: Optional[dict] = None):
        super().__init__(init_cfg=init_cfg)
        # setting of inverted residual blocks
        self.cfgs = self.arch_settings[arch]
        self.img_size = to_2tuple(img_size)

        if isinstance(out_indices, int):
            out_indices = [out_indices]
        assert isinstance(out_indices, Sequence), \
            f'"out_indices" must by a sequence or int, ' \
            f'get {type(out_indices)} instead.'
        for i, index in enumerate(out_indices):
            if index < 0:
                out_indices[i] = 4 + index
                assert out_indices[i] >= 0, f'Invalid out_indices {index}'
        self.out_indices = out_indices

        # building first layer
        input_channel = self.cfgs[0][2]
        patch_embed = torch.nn.Sequential(
            Conv2d_BN(in_channels, input_channel // 2, 3, 2, 1), nn.GELU(),
            Conv2d_BN(input_channel // 2, input_channel, 3, 2, 1))
        layers = [patch_embed]
        # building inverted residual blocks
        block = RepViTBlock

        self.stage_idx = []
        prev_c = input_channel
        for idx, (k, t, c, use_se, use_hs, s) in enumerate(self.cfgs):
            output_channel = make_divisible(c, 8)
            exp_size = make_divisible(input_channel * t, 8)
            if c != prev_c:
                self.stage_idx.append(idx - 1)
                prev_c = c
            layers.append(
                block(input_channel, exp_size, output_channel, k, s, use_se,
                      use_hs))
            input_channel = output_channel
        self.stage_idx.append(idx)
        self.features = nn.ModuleList(layers)
        self.num_stages = len(self.stage_idx)

        stage2_channels = make_divisible(self.cfgs[self.stage_idx[2]][2], 8)
        stage3_channels = make_divisible(self.cfgs[self.stage_idx[3]][2], 8)
        self.out_channels = out_channels
        if self.out_channels > 0:
            self.fuse_stage2 = nn.Conv2d(stage2_channels,
                                         self.out_channels,
                                         kernel_size=1,
                                         bias=False)
            self.fuse_stage3 = nn.Sequential(
                nn.Conv2d(stage3_channels,
                          self.out_channels,
                          kernel_size=1,
                          bias=False),
                nn.Upsample(scale_factor=2, mode=interpolate_mode),
            )

            self.neck = nn.Sequential(
                nn.Conv2d(self.out_channels,
                          self.out_channels,
                          kernel_size=1,
                          bias=False), LayerNorm2d(self.out_channels),
                nn.Conv2d(self.out_channels,
                          self.out_channels,
                          kernel_size=3,
                          padding=1,
                          bias=False), LayerNorm2d(self.out_channels))

        # freeze stages only when self.frozen_stages > 0
        self.frozen_stages = frozen_stages
        if self.frozen_stages > 0:
            self._freeze_stages()

    def forward(self, x):
        counter = 0
        output_dict = dict()
        # patch_embed
        x = self.features[0](x)
        output_dict['stem'] = x

        outs = []
        # stages
        for idx, f in enumerate(self.features[1:]):
            x = f(x)
            if idx in self.stage_idx:
                output_dict[f'stage{counter}'] = x
                counter += 1

                stage_idx = self.stage_idx.index(idx)
                if stage_idx in self.out_indices:
                    if stage_idx < 3:
                        outs.append(x)
                    else:
                        if self.out_channels > 0:
                            x = self.fuse_stage2(
                                output_dict['stage2']) + self.fuse_stage3(
                                    output_dict['stage3'])
                            x = self.neck(x)
                            outs.append(x)

        return tuple(outs)

    def _freeze_stages(self):
        # freeze patch embed
        m = self.features[0]
        m.eval()
        for param in m.parameters():
            param.requires_grad = False

        # freeze layers
        for i in range(self.frozen_stages):
            for j in range(self.stage_idx[i] + 1):
                m = self.features[j + 1]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

        # freeze channel_reduction module
        if self.frozen_stages == self.num_stages and self.out_channels > 0:
            for name in ['fuse_stage2', 'fuse_stage3', 'neck']:
                m = getattr(self, name)
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False
