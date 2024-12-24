import torch
import numpy as np
import time
from typing import Optional
import torch.nn as nn
from mmcv.cnn import build_norm_layer, build_activation_layer
from mmengine.model import BaseModule

class CAA3D(BaseModule):
    """Context Anchor Attention for 3D data."""
    def __init__(
            self,
            channels: int,
            h_kernel_size: int = 11,
            v_kernel_size: int = 11,
            d_kernel_size: int = 11,
            norm_cfg: Optional[dict] = dict(type='BN3d', momentum=0.03, eps=0.001),
            act_cfg: Optional[dict] = dict(type='SiLU'),
            init_cfg: Optional[dict] = None,
    ):
        super().__init__(init_cfg)
        self.avg_pool = nn.AvgPool3d(kernel_size=7, stride=1, padding=3)
        self.conv1 = Conv3dModule(
            channels, channels, kernel_size=1, stride=1, padding=0,
            norm_cfg=norm_cfg, act_cfg=act_cfg
        )
        self.h_conv = Conv3dModule(
            channels, channels, kernel_size=(1, h_kernel_size, 1), stride=1,
            padding=(0, h_kernel_size // 2, 0), groups=channels,
            norm_cfg=None, act_cfg=None
        )
        self.v_conv = Conv3dModule(
            channels, channels, kernel_size=(v_kernel_size, 1, 1), stride=1,
            padding=(v_kernel_size // 2, 0, 0), groups=channels,
            norm_cfg=None, act_cfg=None
        )
        self.d_conv = Conv3dModule(
            channels, channels, kernel_size=(1, 1, d_kernel_size), stride=1,
            padding=(0, 0, d_kernel_size // 2), groups=channels,
            norm_cfg=None, act_cfg=None
        )
        self.conv2 = Conv3dModule(
            channels, channels, kernel_size=1, stride=1, padding=0,
            norm_cfg=norm_cfg, act_cfg=act_cfg
        )
        self.act = nn.Sigmoid()
        self.norm = nn.LayerNorm(channels)


    def forward(self, x):
        print(f"Input shape: {x.shape}")
        # 3D average pooling -> conv1 -> horizontal conv -> vertical conv -> depth conv -> conv2 -> sigmoid
        x0 =self.avg_pool(x)
        x1 = self.conv1(x0)
        x2 = self.h_conv(x1)
        x3 = self.v_conv(x2)
        x4 = self.d_conv(x3)
        x5 = self.conv2(x4)
        x6 = x5.flatten(2).transpose(1, 2)
        x7 = self.norm(x6)
        output = self.act(x7)
        print(f"Attention factor shape: {output.shape}")
        return output


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
    device = torch.device("cpu")

    # Input tensor
    input = torch.randint(
        low=0,
        high=255,
        size=(56,32, 24, 24, 24),
        dtype=torch.float,
    ).to(device)

    # Initialize model
    model = CAA3D(channels=32).to(device)

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
