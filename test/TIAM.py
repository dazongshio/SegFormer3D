import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from monai.networks.blocks.convolutions import Convolution
from monai.networks.layers.factories import Act, Norm
from monai.networks.layers.utils import get_act_layer, get_norm_layer

# 论文：Robust change detection for remote sensing images based on temporospatial interactive attention module
# 论文地址：https://www.sciencedirect.com/science/article/pii/S1569843224001213

class SpatiotemporalAttentionFull(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=2, sub_sample=False):
        super(SpatiotemporalAttentionFull, self).__init__()
        assert dimension in [2, ]
        self.dimension = dimension
        self.sub_sample = sub_sample
        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        self.g = nn.Sequential(
            nn.BatchNorm3d(self.in_channels),
            nn.Conv3d(in_channels=self.in_channels, out_channels=self.inter_channels,
                      kernel_size=1, stride=1, padding=0)
        )

        self.W = nn.Sequential(
            nn.Conv3d(in_channels=self.inter_channels, out_channels=self.in_channels,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm3d(self.in_channels)
        )
        self.theta = nn.Sequential(
            nn.BatchNorm3d(self.in_channels),
            nn.Conv3d(in_channels=self.in_channels, out_channels=self.inter_channels,
                      kernel_size=1, stride=1, padding=0),
        )
        self.phi = nn.Sequential(
            nn.BatchNorm3d(self.in_channels),
            nn.Conv3d(in_channels=self.in_channels, out_channels=self.inter_channels,
                      kernel_size=1, stride=1, padding=0),
        )


        self.energy_time_1_sf = nn.Softmax(dim=-1)
        self.energy_time_2_sf = nn.Softmax(dim=-1)
        self.energy_space_2s_sf = nn.Softmax(dim=-2)
        self.energy_space_1s_sf = nn.Softmax(dim=-2)

        self.GN = nn.GroupNorm(num_groups=8, num_channels=self.in_channels)
        self.transp_conv = get_conv_layer(
            spatial_dims= 3,
            in_channels = self.inter_channels,
            out_channels = self.in_channels,
            kernel_size=2,
            stride=2,
            dropout=0.,
            bias=False,
            conv_only=True,
            is_transposed=True,
        )

    def forward(self, x2, x1):
        # 从第一个输入张量中获取批量大小
        batch_size = x1.size(0)
        x2 = self.transp_conv(x2)
        x2 = self.GN(x2)

        # 对 x1 应用 g 变换并重塑以便进一步计算
        g_x11 = self.g(x1).reshape(batch_size, self.inter_channels, -1)  # 形状: [batch_size, inter_channels, H*W*D]
        # Todo 有时间修改一下，看看这里遗漏了什么
        g_x12 = g_x11.permute(0, 2, 1)  # 转置为形状: [batch_size, H*W*D, inter_channels]

        # 对 x2 应用 g 变换并重塑
        g_x21 = self.g(x2).reshape(batch_size, self.inter_channels, -1)  # 形状: [batch_size, inter_channels, H*W*D]
        g_x22 = g_x21.permute(0, 2, 1)  # 转置为形状: [batch_size, H*W*D, inter_channels]

        # 对 x1 应用 theta 变换并重塑
        theta_x1 = self.theta(x1).reshape(batch_size, self.inter_channels, -1)  # 形状: [batch_size, inter_channels, H*W*D]
        theta_x2 = theta_x1.permute(0, 2, 1)  # 转置为形状: [batch_size, H*W*D, inter_channels]

        # 对 x2 应用 phi 变换并重塑
        phi_x1 = self.phi(x2).reshape(batch_size, self.inter_channels, -1)  # 形状: [batch_size, inter_channels, H*W*D]
        phi_x2 = phi_x1.permute(0, 2, 1)  # 转置为形状: [batch_size, H*W*D, inter_channels]

        # 计算时序和空间关系的能量矩阵
        energy_time_1 = torch.matmul(theta_x1, phi_x2)  # 形状: [batch_size, inter_channels, inter_channels]
        energy_time_2 = energy_time_1.permute(0, 2, 1)  # 转置时序能量矩阵
        energy_space_1 = torch.matmul(theta_x2, phi_x1)  # 形状: [batch_size, H*W*D, H*W*D]
        energy_space_2 = energy_space_1.permute(0, 2, 1)  # 转置空间能量矩阵

        # 对能量矩阵应用缩放函数
        energy_time_1s = self.energy_time_1_sf(energy_time_1)
        energy_time_2s = self.energy_time_2_sf(energy_time_2)
        energy_space_2s = self.energy_space_2s_sf(energy_space_2)
        energy_space_1s = self.energy_space_2s_sf(energy_space_1)

        # 使用缩放后的能量矩阵重建表示
        # y1: 基于与 x2 的时空交互重建 x1
        # energy_time_2s * g_x11 * energy_space_2s 对应:
        # C2*S(C1) x C1*H1W1 x S(H1W1)*H2W2 -> 形状: [batch_size, C2, H2*W2]
        y1 = torch.matmul(torch.matmul(energy_time_2s, g_x11), energy_space_2s).contiguous()

        # y2: 基于与 x1 的时空交互重建 x2
        # energy_time_1s * g_x21 * energy_space_1s 对应:
        # C1*S(C2) x C2*H2W2 x S(H2W2)*H1W1 -> 形状: [batch_size, C1, H1*W1]
        y2 = torch.matmul(torch.matmul(energy_time_1s, g_x21), energy_space_1s).contiguous()

        # 将 y1 和 y2 重塑回与 x2 和 x1 分别匹配的空间维度
        y1 = y1.reshape(batch_size, self.inter_channels, *x2.size()[2:])
        y2 = y2.reshape(batch_size, self.inter_channels, *x1.size()[2:])

        # 应用 W 变换并将结果加回到原始输入 x1 和 x2
        # Todo添加了加号，不知道好与坏

        return x1 + self.W(y1)+ x2 + self.W(y2)


class SpatiotemporalAttentionBase(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=2, sub_sample=False):
        super(SpatiotemporalAttentionBase, self).__init__()
        assert dimension in [2, ]
        self.dimension = dimension
        self.sub_sample = sub_sample
        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        self.g = nn.Sequential(
            nn.BatchNorm2d(self.in_channels),
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                      kernel_size=1, stride=1, padding=0)
        )

        self.W = nn.Sequential(
            nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.in_channels)
        )

        self.theta = nn.Sequential(
            nn.BatchNorm2d(self.in_channels),
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                      kernel_size=1, stride=1, padding=0),
        )
        self.phi = nn.Sequential(
            nn.BatchNorm2d(self.in_channels),
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                      kernel_size=1, stride=1, padding=0),
        )
        self.energy_space_2s_sf = nn.Softmax(dim=-2)
        self.energy_space_1s_sf = nn.Softmax(dim=-2)

    def forward(self, x1, x2):
        """
        :param x: (b, c, h, w)
        :param return_nl_map: if True return z, nl_map, else only return z.
        :return:
        """
        batch_size = x1.size(0)
        g_x11 = self.g(x1).reshape(batch_size, self.inter_channels, -1)
        g_x21 = self.g(x2).reshape(batch_size, self.inter_channels, -1)

        theta_x1 = self.theta(x1).reshape(batch_size, self.inter_channels, -1)
        theta_x2 = theta_x1.permute(0, 2, 1)

        phi_x1 = self.phi(x2).reshape(batch_size, self.inter_channels, -1)

        energy_space_1 = torch.matmul(theta_x2, phi_x1)
        energy_space_2 = energy_space_1.permute(0, 2, 1)
        energy_space_2s = self.energy_space_2s_sf(energy_space_1)  # S(H1W1)*H2W2
        energy_space_1s = self.energy_space_1s_sf(energy_space_2)  # S(H2W2)*H1W1

        # g_x11*energy_space_2s = C1*H1W1 × S(H1W1)*H2W2 = (C1*H2W2)' is rebuild C1*H1W1
        y1 = torch.matmul(g_x11, energy_space_2s).contiguous()  # C2*H2W2
        # g_x21*energy_space_1s = C2*H2W2 × S(H2W2)*H1W1 = (C2*H1W1)' is rebuild C2*H2W2
        y2 = torch.matmul(g_x21, energy_space_1s).contiguous()
        y1 = y1.reshape(batch_size, self.inter_channels, *x2.size()[2:])
        y2 = y2.reshape(batch_size, self.inter_channels, *x1.size()[2:])
        return x1 + self.W(y1), x2 + self.W(y2)


class SpatiotemporalAttentionFullNotWeightShared(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=2, sub_sample=False):
        super(SpatiotemporalAttentionFullNotWeightShared, self).__init__()
        assert dimension in [2, ]
        self.dimension = dimension
        self.sub_sample = sub_sample
        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        self.g1 = nn.Sequential(
            nn.BatchNorm2d(self.in_channels),
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                      kernel_size=1, stride=1, padding=0)
        )
        self.g2 = nn.Sequential(
            nn.BatchNorm2d(self.in_channels),
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                      kernel_size=1, stride=1, padding=0),
        )

        self.W1 = nn.Sequential(
            nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.in_channels)
        )
        self.W2 = nn.Sequential(
            nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.in_channels)
        )
        self.theta = nn.Sequential(
            nn.BatchNorm2d(self.in_channels),
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                      kernel_size=1, stride=1, padding=0),
        )
        self.phi = nn.Sequential(
            nn.BatchNorm2d(self.in_channels),
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                      kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x1, x2):
        """
        :param x: (b, c, h, w)
        :param return_nl_map: if True return z, nl_map, else only return z.
        :return:
        """
        batch_size = x1.size(0)
        g_x11 = self.g1(x1).reshape(batch_size, self.inter_channels, -1)
        g_x12 = g_x11.permute(0, 2, 1)
        g_x21 = self.g2(x2).reshape(batch_size, self.inter_channels, -1)
        g_x22 = g_x21.permute(0, 2, 1)

        theta_x1 = self.theta(x1).reshape(batch_size, self.inter_channels, -1)
        theta_x2 = theta_x1.permute(0, 2, 1)

        phi_x1 = self.phi(x2).reshape(batch_size, self.inter_channels, -1)
        phi_x2 = phi_x1.permute(0, 2, 1)

        energy_time_1 = torch.matmul(theta_x1, phi_x2)
        energy_time_2 = energy_time_1.permute(0, 2, 1)
        energy_space_1 = torch.matmul(theta_x2, phi_x1)
        energy_space_2 = energy_space_1.permute(0, 2, 1)

        energy_time_1s = F.softmax(energy_time_1, dim=-1)
        energy_time_2s = F.softmax(energy_time_2, dim=-1)
        energy_space_2s = F.softmax(energy_space_1, dim=-2)
        energy_space_1s = F.softmax(energy_space_2, dim=-2)
        # C1*S(C2) energy_time_1s * C1*H1W1 g_x12 * energy_space_1s S(H2W2)*H1W1 -> C1*H1W1
        y1 = torch.matmul(torch.matmul(energy_time_2s, g_x11), energy_space_2s).contiguous()  # C2*H2W2
        # C2*S(C1) energy_time_2s * C2*H2W2 g_x21 * energy_space_2s S(H1W1)*H2W2 -> C2*H2W2
        y2 = torch.matmul(torch.matmul(energy_time_1s, g_x21), energy_space_1s).contiguous()  # C1*H1W1
        y1 = y1.reshape(batch_size, self.inter_channels, *x2.size()[2:])
        y2 = y2.reshape(batch_size, self.inter_channels, *x1.size()[2:])
        return x1 + self.W1(y1), x2 + self.W2(y2)


class UnetOutBlock(nn.Module):
    def __init__(
        self, spatial_dims: int, in_channels: int, out_channels: int, dropout: Optional[Union[Tuple, str, float]] = None
    ):
        super().__init__()
        self.conv = get_conv_layer(
            spatial_dims, in_channels, out_channels, kernel_size=1, stride=1, dropout=dropout, bias=True, conv_only=True
        )

    def forward(self, inp):
        return self.conv(inp)


def get_conv_layer(
    spatial_dims: int,
    in_channels: int,
    out_channels: int,
    kernel_size: Union[Sequence[int], int] = 3,
    stride: Union[Sequence[int], int] = 1,
    act: Optional[Union[Tuple, str]] = Act.PRELU,
    norm: Union[Tuple, str] = Norm.INSTANCE,
    dropout: Optional[Union[Tuple, str, float]] = None,
    bias: bool = False,
    conv_only: bool = True,
    is_transposed: bool = False,
):
    padding = get_padding(kernel_size, stride)
    output_padding = None
    if is_transposed:
        output_padding = get_output_padding(kernel_size, stride, padding)
    return Convolution(
        spatial_dims,
        in_channels,
        out_channels,
        strides=stride,
        kernel_size=kernel_size,
        act=act,
        norm=norm,
        dropout=dropout,
        bias=bias,
        conv_only=conv_only,
        is_transposed=is_transposed,
        padding=padding,
        output_padding=output_padding,
    )


def get_padding(
    kernel_size: Union[Sequence[int], int], stride: Union[Sequence[int], int]
) -> Union[Tuple[int, ...], int]:

    kernel_size_np = np.atleast_1d(kernel_size)
    stride_np = np.atleast_1d(stride)
    padding_np = (kernel_size_np - stride_np + 1) / 2
    if np.min(padding_np) < 0:
        raise AssertionError("padding value should not be negative, please change the kernel size and/or stride.")
    padding = tuple(int(p) for p in padding_np)

    return padding if len(padding) > 1 else padding[0]


def get_output_padding(
    kernel_size: Union[Sequence[int], int], stride: Union[Sequence[int], int], padding: Union[Sequence[int], int]
) -> Union[Tuple[int, ...], int]:
    kernel_size_np = np.atleast_1d(kernel_size)
    stride_np = np.atleast_1d(stride)
    padding_np = np.atleast_1d(padding)

    out_padding_np = 2 * padding_np + stride_np - kernel_size_np
    if np.min(out_padding_np) < 0:
        raise AssertionError("out_padding value should not be negative, please change the kernel size and/or stride.")
    out_padding = tuple(int(p) for p in out_padding_np)

    return out_padding if len(out_padding) > 1 else out_padding[0]
if __name__ == '__main__':
    device = torch.device("cuda:1")
    input1 = torch.randn(56, 32, 24, 24, 24).to(device)  # B C H W
    input2 = torch.randn(56, 64, 12, 12, 12).to(device)  # B C H W
    input3 = torch.randn(56, 160, 6, 6, 6).to(device)  # B C H W
    input4 = torch.randn(56, 256, 3, 3, 3).to(device)  # B C H W


    sp43 = SpatiotemporalAttentionFull(160,256).to(device)
    x3 = sp43(input4, input3)
    sp32 = SpatiotemporalAttentionFull(64,160).to(device)
    x2 = sp32(x3, input2)
    sp21 = SpatiotemporalAttentionFull(32,64).to(device)
    # x1 = sp21(input2, input1)

    # output_full_x1, output_full_x2, output_full_x3, output_full_x4, = sp_full(input1, input2, input3, input4)
    #
    print(input1.shape, input2.shape)
    # x1 = sp21(x2, input1)
    # print(output_full_x1.shape, output_full_x2.shape)

    sp_base = SpatiotemporalAttentionBase(in_channels=32)
    output_base_x1, output_base_x2 = sp_base(input1, input2)

    print(input1.shape, input2.shape)
    print(output_base_x1.shape, output_base_x2.shape)

    sp_full_not_shared = SpatiotemporalAttentionFullNotWeightShared(in_channels=64)
    output_full_not_shared_x1, output_full_not_shared_x2 = sp_full_not_shared(input1, input2)

    print(input1.shape, input2.shape)
    print(output_full_not_shared_x1.shape, output_full_not_shared_x2.shape)
