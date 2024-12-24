import torch
import torch.nn as nn
from torch.nn import Softmax

# 定义一个无限小的矩阵，用于在注意力矩阵中屏蔽特定位置
def INF(B, H, W):
    return -torch.diag(torch.tensor(float("inf")).repeat(H), 0).unsqueeze(0).repeat(B * W, 1, 1)

class CrissCrossAttention(nn.Module):
    """ Criss-Cross Attention Module"""
    def __init__(self, in_dim):
        super(CrissCrossAttention, self).__init__()
        # Q, K, V转换层
        self.query_conv = nn.Linear(in_dim, in_dim, bias=False)
        self.key_conv = nn.Linear(in_dim, in_dim, bias=False)
        self.value_conv = nn.Linear(in_dim, in_dim, bias=False)
        # 使用softmax对注意力分数进行归一化
        self.softmax = Softmax(dim=3)
        self.INF = INF
        # 学习一个缩放参数，用于调节注意力的影响
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # (batch_size, num_patches, hidden_size)
        B, N, C = x.shape
        m_batchsize, _, height, width = x.size()
        # 计算查询(Q)、键(K)、值(V)矩阵
        proj_query = self.query_conv(x).reshape(B, N, self.num_heads, self.attention_head_dim).permute(0, 2, 1, 3)
        proj_query = proj_query.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height).permute(0, 2, 1)

        q = (
            self.query(x)
            .reshape(B, N, self.num_heads, self.attention_head_dim)
            .permute(0, 2, 1, 3)
        )

        proj_key = self.key_conv(x)
        proj_key = proj_key.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)

        proj_value = self.value_conv(x)
        proj_value = proj_value.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)

        # 计算垂直和水平方向上的注意力分数，并应用无穷小掩码屏蔽自注意
        energy = (torch.bmm(proj_query, proj_key) + self.INF(m_batchsize, height, width)).view(m_batchsize, width, height, height).permute(0, 2, 1, 3)

        # 在垂直和水平方向上应用softmax归一化
        concate = self.softmax(energy)

        # 分离垂直和水平方向上的注意力，应用到值(V)矩阵上
        att = concate[:, :, :, 0:height].permute(0, 2, 1, 3).contiguous().view(m_batchsize * width, height, height)

        # 计算最终的输出，加上输入x以应用残差连接
        out = torch.bmm(proj_value, att.permute(0, 2, 1)).view(m_batchsize, width, -1, height).permute(0, 2, 3, 1)

        return self.gamma * (out) + x

if __name__ == '__main__':
    block = CrissCrossAttention(64)
    input = torch.rand(56, 13824, 4)
    output = block(input)
    print( output.shape) # 打印输出形状