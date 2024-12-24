import torch
import numpy as np
import time
import torch.nn as nn
from mmcv.cnn import build_activation_layer
from typing import Optional
from mmcv.cnn import ConvModule, build_norm_layer
from mmengine.model import BaseModule


# 论文源码地址 https://github.com/NUST-Machine-Intelligence-Laboratory/PKINet
class CAA(BaseModule):
    """Context Anchor Attention"""

    def __init__(
            self,
            channels: int,
            h_kernel_size: int = 11,
            v_kernel_size: int = 11,
            norm_cfg: Optional[dict] = dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg: Optional[dict] = dict(type='SiLU'),
            init_cfg: Optional[dict] = None,
    ):
        super().__init__(init_cfg)
        self.avg_pool = nn.AvgPool2d(7, 1, 3)
        self.conv1 = ConvModule(channels, channels, 1, 1, 0,
                                norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.h_conv = ConvModule(channels, channels, (1, h_kernel_size), 1,
                                 (0, h_kernel_size // 2), groups=channels,
                                 norm_cfg=None, act_cfg=None)
        self.v_conv = ConvModule(channels, channels, (v_kernel_size, 1), 1,
                                 (v_kernel_size // 2, 0), groups=channels,
                                 norm_cfg=None, act_cfg=None)
        self.conv2 = ConvModule(channels, channels, 1, 1, 0,
                                norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.act = nn.Sigmoid()

    def forward(self, x):
        print(f"Input shape: {x.shape}")
        attn_factor = self.act(self.conv2(self.v_conv(self.h_conv(self.conv1(self.avg_pool(x))))))
        print(f"Attention factor shape: {attn_factor.shape}")
        return attn_factor


class CAAModule(nn.Module):
    def __init__(self, channels: int, h_kernel_size: int = 11, v_kernel_size: int = 11):
        super().__init__()
        self.caa = CAA(channels, h_kernel_size, v_kernel_size)

    def forward(self, x):
        return self.caa(x)


class Conv3dModule(nn.Module):
    """3D Convolution Module with optional normalization and activation."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 groups=1, norm_cfg=None, act_cfg=None):
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels, out_channels, kernel_size, stride, padding, groups=groups
        )
        self.norm = build_norm_layer(norm_cfg, out_channels)[1] if norm_cfg else None
        self.activate = build_activation_layer(act_cfg) if act_cfg else None

    def forward(self, x):
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.activate:
            x = self.activate(x)
        return x

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Input tensor
    input = torch.randint(
        low=0,
        high=255,
        size=(4, 96, 96, 96),
        dtype=torch.float,
    ).to(device)

    # Initialize model
    model = CAAModule(channels=96).to(device)

    # Measure time
    start = time.time()
    output = model(input)

    # Print output shape
    print(f"Output shape: {output.shape}")

    # Count trainable parameters
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print('The number of params in Million: ', params / 1e6)
    print(f"Execution time: {time.time() - start:.2f} s")
