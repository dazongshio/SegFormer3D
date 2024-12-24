import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.xpu import device
from einops import rearrange, repeat


# Unfold模块使用给定的kernel_size对输入进行展开
class Unfold(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        # kernel_size定义了展开操作的窗口大小
        self.kernel_size = kernel_size
        # 初始化权重为单位矩阵，使得每个窗口内的元素直接复制到输出
        weights = torch.eye(kernel_size)
        weights = weights.reshape(kernel_size, 1, kernel_size)
        # 将权重设置为不需要梯度，因为它们不会在训练过程中更新
        self.weights = nn.Parameter(weights, requires_grad=False)

    def forward(self, x):  # 获取输入的批量大小、通道数、高度和宽度
        b, c, n = x.shape
        # 使用定义好的权重对输入进行卷积操作，实现展开功能
        x = F.conv1d(x.reshape(b * c, 1, n), self.weights, stride=1, padding=self.kernel_size // 2)
        # 调整输出的形状，使其包含展开的窗口
        return x.reshape(b, c * 3, n)


# Fold模块与Unfold相反，用于将展开的特征图折叠回原始形状
class Fold(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()

        self.kernel_size = kernel_size
        # 与Unfold相同，初始化权重为单位矩阵
        weights = torch.eye(kernel_size)
        weights = weights.reshape(kernel_size, 1, kernel_size)
        # 权重不需要梯度
        self.weights = nn.Parameter(weights, requires_grad=False)

    def forward(self, x):
        b, _, n = x.shape
        # 使用转置卷积（逆卷积）操作恢复原始大小的特征图
        x = F.conv_transpose1d(x, self.weights, stride=1, padding=self.kernel_size // 2)
        return x


# Attention模块实现自注意力机制
class Attention(nn.Module):
    def __init__(self, dim, window_size=None, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()

        self.dim = dim  # dim定义了特征维度，num_heads定义了注意力头的数量
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.window_size = window_size
        # 根据给定的尺度因子或自动计算的尺度进行缩放
        self.scale = qk_scale or head_dim ** -0.5
        # qkv用一个卷积层同时生成查询、键和值
        self.qkv = nn.Conv1d(dim, dim * 3, 1, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv1d(dim, dim, 1)  # proj是输出的投影层
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        # B, C, H, W = x.shape # 获取输入的形状
        # N = H * W
        B, C, N = x.shape
        # 将qkv的输出重塑为适合自注意力计算的形状
        q, k, v = self.qkv(x).reshape(B, self.num_heads, C // self.num_heads * 3, N).chunk(3,
                                                                                           dim=2)  # (B, num_heads, head_dim, N)
        # 计算注意力分数，注意力分数乘以尺度因子
        attn = (k.transpose(-1, -2) @ q) * self.scale
        # 应用softmax获取注意力权重
        attn = attn.softmax(dim=-1)  # (B, h, N, N)
        # 应用注意力dropout
        attn = self.attn_drop(attn)

        x = (v @ attn)

        x = self.proj(rearrange(x, 'b c d n-> b (c d) n'))
        x = self.proj_drop(x)
        return x


# StokenAttention模块通过迭代地细化空间Token以增强特征表示
class StokenAttention(nn.Module):
    def __init__(self, dim, stoken_size, n_iter=1, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
                 proj_drop=0.):
        super().__init__()
        self.n_iter = n_iter
        self.stoken_size = stoken_size
        self.scale = dim ** - 0.5
        self.unfold = Unfold(3)
        self.fold = Fold(3)
        self.stoken_refine = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                       attn_drop=attn_drop, proj_drop=proj_drop)

    def stoken_forward(self, x):
        '''
           x: (B, C, N)
        '''
        x = x.permute(0, 2, 1)
        B, C, N0 = x.shape
        n = self.stoken_size
        # 计算padding
        pad_left = 0  # 通常左侧填充为 0
        pad_right = (n - N0 % n) % n  # 计算右侧填充量，使得长度变为 l 的整数倍

        # 应用填充
        if pad_right > 0:
            x = F.pad(x, (pad_left, pad_right))  # 填充 1D 张量

        B, C, N = x.shape
        # _, _, H, W = x.shape
        nn = N // n

        # 使用自适应平均池化得到空间Token的特征
        stoken_features = F.adaptive_avg_pool1d(x, nn)  # (B, C, nn)
        # 展开特征以进行精细化处理
        pixel_features = x.reshape(B, C, nn, n).permute(0, 2, 3, 1).reshape(B, nn, n, C)
        # 使用没有梯度的操作进行迭代精细化
        with torch.no_grad():
            for idx in range(self.n_iter):
                stoken_features = self.unfold(stoken_features)  # (B, C*9, hh*ww)
                stoken_features = stoken_features.transpose(1, 2).reshape(B, nn, C, 3)
                affinity_matrix = pixel_features @ stoken_features * self.scale  # (B, hh*ww, h*w, 9)

                affinity_matrix = affinity_matrix.softmax(-1)  # (B, hh*ww, h*w, 9)

                affinity_matrix_sum = affinity_matrix.sum(2).transpose(1, 2).reshape(B, 3, nn)

                affinity_matrix_sum = self.fold(affinity_matrix_sum)
                if idx < self.n_iter - 1:
                    stoken_features = pixel_features.transpose(-1, -2) @ affinity_matrix  # (B, hh*ww, C, 9)

                    stoken_features = self.fold(stoken_features.permute(0, 2, 3, 1).reshape(B * C, 3, nn)).reshape(
                        B, C, nn)

                    stoken_features = stoken_features / (affinity_matrix_sum + 1e-12)  # (B, C, hh, ww)

        stoken_features = pixel_features.transpose(-1, -2) @ affinity_matrix  # (B, hh*ww, C, 9)

        stoken_features = self.fold(stoken_features.permute(0, 2, 3, 1).reshape(B * C, 3, nn)).reshape(B, C, nn)

        stoken_features = stoken_features / (affinity_matrix_sum.detach() + 1e-12)  # (B, C, hh, ww)

        stoken_features = self.stoken_refine(stoken_features)

        stoken_features = self.unfold(stoken_features)  # (B, C*9, hh*ww)
        stoken_features = stoken_features.transpose(1, 2).reshape(B, nn, C, 3)  # (B, hh*ww, C, 9)
        # 通过affinity_matrix将精细化的特征映射回原始像素级别
        pixel_features = stoken_features @ affinity_matrix.transpose(-1, -2)  # (B, hh*ww, C, h*w)
        # 折叠特征，恢复原始形状
        pixel_features = pixel_features.reshape(B, nn, C, n).permute(0, 2, 1, 3).reshape(B, C, N)

        if pad_right > 0:
            pixel_features = pixel_features[:, :, :N0]

        return pixel_features

    def direct_forward(self, x):  # 直接对x应用Attention进行细化
        B, N, C = x.shape
        stoken_features = x.permute(0, 2, 1)
        stoken_features = self.stoken_refine(stoken_features)
        return stoken_features

    def forward(self, x):
        if self.stoken_size > 1:
            return self.stoken_forward(x)
        else:
            return self.direct_forward(x)


#  输入 N C H W,  输出 N C H W
if __name__ == '__main__':
    device = torch.device("cuda:1")
    input = torch.randn(56, 13824, 32).to(device)  # 创建一个随机输入
    se = StokenAttention(32, stoken_size=8).to(device)  # 实例化注意力模块
    output = se(input)
    print(output.shape)  # 打印输出形状
