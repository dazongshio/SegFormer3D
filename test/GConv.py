import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
import torch.fft
from torch.nn import LayerNorm


# 定义一个函数用来获取深度可分离卷积层
def get_dwconv(dim, kernel, bias):
    return nn.Conv2d(dim, dim, kernel_size=kernel, padding=(kernel - 1) // 2, bias=bias, groups=dim)

# 全局和局部过滤器模块
class GlobalLocalFilter(nn.Module):
    def __init__(self, dim, h=14, w=8):
        super().__init__()
        # 使用深度可分离卷积处理一半的通道
        self.dw = nn.Conv2d(dim // 2, dim // 2, kernel_size=3, padding=1, bias=False, groups=dim // 2)
        # 初始化一个复数权重参数，用于后续的频域操作
        self.complex_weight = nn.Parameter(torch.randn(dim // 2, h, w, 2, dtype=torch.float32) * 0.02)
        trunc_normal_(self.complex_weight, std=.02)
        # 定义输入和输出前的层归一化
        self.pre_norm = LayerNorm(dim, eps=1e-6, data_format='channels_first')
        self.post_norm = LayerNorm(dim, eps=1e-6, data_format='channels_first')

    def forward(self, x):
        x = self.pre_norm(x)
        # 将输入分成两部分
        x1, x2 = torch.chunk(x, 2, dim=1)
        # 对第一部分应用深度可分离卷积
        x1 = self.dw(x1)
        # 对第二部分进行傅里叶变换
        x2 = x2.to(torch.float32)
        B, C, a, b = x2.shape
        x2 = torch.fft.rfft2(x2, dim=(2, 3), norm='ortho')
        weight = self.complex_weight
        # 如果权重形状和x2形状不匹配，则调整权重大小
        if not weight.shape[1:3] == x2.shape[2:4]:
            weight = F.interpolate(weight.permute(3, 0, 1, 2), size=x2.shape[2:4], mode='bilinear', align_corners=True).permute(1, 2, 3, 0)
        weight = torch.view_as_complex(weight.contiguous())
        # 应用复数权重并进行逆傅里叶变换
        x2 = x2 * weight
        x2 = torch.fft.irfft2(x2, s=(a, b), dim=(2, 3), norm='ortho')
        # 将两部分的结果合并并通过层归一化
        x = torch.cat([x1.unsqueeze(2), x2.unsqueeze(2)], dim=2).reshape(B, 2 * C, a, b)
        x = self.post_norm(x)
        return x

# 高阶非局部卷积模块
class gnconv(nn.Module):
    def __init__(self, dim, order=3, gflayer=None, h=14, w=8, s=1.0):
        super().__init__()
        self.order = order
        # 根据高阶数生成不同尺寸的维度列表
        self.dims = [dim // 2 ** i for i in range(order)]
        self.dims.reverse()
        self.proj_in = nn.Conv2d(dim, 2 * dim, 1)# 输入投影卷积

        if gflayer is None:
            self.dwconv = get_dwconv(sum(self.dims), 7, True)# 获取深度可分离卷积
        else:
            self.dwconv = gflayer(sum(self.dims), h=h, w=w)# 或者使用自定义的全局局部过滤器
        self.proj_out = nn.Conv2d(dim, dim, 1)
        self.pws = nn.ModuleList(
            [nn.Conv2d(self.dims[i], self.dims[i + 1], 1) for i in range(order - 1)]
        )
        self.scale = s
        print('[gnconv]', order, 'order with dims=', self.dims, 'scale=%.4f' % self.scale)

    def forward(self, x):
        fused_x = self.proj_in(x)
        pwa, abc = torch.split(fused_x, (self.dims[0], sum(self.dims)), dim=1)
        dw_abc = self.dwconv(abc) * self.scale
        dw_list = torch.split(dw_abc, self.dims, dim=1)
        x = pwa * dw_list[0]
        for i in range(self.order - 1):
            x = self.pws[i](x) * dw_list[i + 1]
        x = self.proj_out(x)
        return x

if __name__ == '__main__':

    a = torch.ones(3, 32, 20, 20)  #生成随机数
    b = gnconv(32)  #实例化
    c = b(a)
    print(c.size())