
import torch.nn as nn
import torch
from einops import rearrange
import math



class AKConv(nn.Module):
    def __init__(self, inc, outc, num_param, stride=1, bias=None):
        super(AKConv, self).__init__()
        self.num_param = num_param
        self.stride = stride
        self.conv = nn.Sequential(nn.Conv3d(inc, outc, kernel_size=(3,3, 3), padding=1,stride=(2,2, 2), bias=bias),
                                  nn.BatchNorm3d(outc),
                                  nn.SiLU())  # the conv adds the BN and SiLU to compare original Conv in YOLOv5.
        self.p_conv = nn.Conv3d(inc, 3 * num_param, kernel_size=3, padding=1, stride=stride)
        nn.init.constant_(self.p_conv.weight, 0)
        self.p_conv.register_full_backward_hook(self._set_lr)

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

    def forward(self, x): # （(56, 32, 24, 24, 24)
        # N is num_param.
        offset = self.p_conv(x) # (56, 32, 24, 24, 24)
        dtype = offset.data.type()# 'torch.FloatTensor'
        N = offset.size(1) // 3 # 64
        # (b, 3N, h, w, d)
        p = self._get_p(offset, dtype)

        # (b, h, w, d, 3N)
        p = p.contiguous().permute(0, 2, 3, 4, 1)
        q_lt = p.detach().floor()
        q_rb = q_lt + 1

        # (b, h, w, d, N) 分别裁剪 x, y, z 坐标范围
        q_lt = torch.cat([
            torch.clamp(q_lt[..., :N], 0, x.size(2) - 1),  # x
            torch.clamp(q_lt[..., N:2 * N], 0, x.size(3) - 1),  # y
            torch.clamp(q_lt[..., 2 * N:], 0, x.size(4) - 1)  # z
        ], dim=-1).long()

        q_rb = torch.cat([
            torch.clamp(q_rb[..., :N], 0, x.size(2) - 1),  # x
            torch.clamp(q_rb[..., N:2 * N], 0, x.size(3) - 1),  # y
            torch.clamp(q_rb[..., 2 * N:], 0, x.size(4) - 1)  # z
        ], dim=-1).long()

        # 生成其他 6 个点：q_lb, q_rt, q_lf, q_rf, q_rb, q_rt
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:2 * N], q_lt[..., 2 * N:]], dim=-1)  # x_lt, y_rb, z_lt
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:2 * N], q_lt[..., 2 * N:]], dim=-1)  # x_rb, y_lt, z_lt
        q_lf = torch.cat([q_lt[..., :N], q_lt[..., N:2 * N], q_rb[..., 2 * N:]], dim=-1)  # x_lt, y_lt, z_rb
        q_rf = torch.cat([q_rb[..., :N], q_rb[..., N:2 * N], q_lt[..., 2 * N:]], dim=-1)  # x_rb, y_rb, z_lt

        # 裁剪 p 的坐标范围
        p = torch.cat([
            torch.clamp(p[..., :N], 0, x.size(2) - 1),  # x
            torch.clamp(p[..., N:2 * N], 0, x.size(3) - 1),  # y
            torch.clamp(p[..., 2 * N:], 0, x.size(4) - 1)  # z
        ], dim=-1)

        # 三线性插值系数
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * \
               (1 + (q_lt[..., N:2 * N].type_as(p) - p[..., N:2 * N])) * \
               (1 + (q_lt[..., 2 * N:].type_as(p) - p[..., 2 * N:]))

        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * \
               (1 - (q_rb[..., N:2 * N].type_as(p) - p[..., N:2 * N])) * \
               (1 - (q_rb[..., 2 * N:].type_as(p) - p[..., 2 * N:]))

        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * \
               (1 - (q_lb[..., N:2 * N].type_as(p) - p[..., N:2 * N])) * \
               (1 + (q_lb[..., 2 * N:].type_as(p) - p[..., 2 * N:]))

        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * \
               (1 + (q_rt[..., N:2 * N].type_as(p) - p[..., N:2 * N])) * \
               (1 + (q_rt[..., 2 * N:].type_as(p) - p[..., 2 * N:]))

        g_lf = (1 + (q_lf[..., :N].type_as(p) - p[..., :N])) * \
               (1 + (q_lf[..., N:2 * N].type_as(p) - p[..., N:2 * N])) * \
               (1 - (q_lf[..., 2 * N:].type_as(p) - p[..., 2 * N:]))

        g_rf = (1 - (q_rf[..., :N].type_as(p) - p[..., :N])) * \
               (1 - (q_rf[..., N:2 * N].type_as(p) - p[..., N:2 * N])) * \
               (1 - (q_rf[..., 2 * N:].type_as(p) - p[..., 2 * N:]))

        # 从原始张量中采样
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)
        x_q_lf = self._get_x_q(x, q_lf, N)
        x_q_rf = self._get_x_q(x, q_rf, N)

        # 三线性插值
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt + \
                   g_lf.unsqueeze(dim=1) * x_q_lf + \
                   g_rf.unsqueeze(dim=1) * x_q_rf

        x_offset = self._reshape_x_offset(x_offset, self.num_param)
        out = self.conv(x_offset)

        return out

    # generating the inital sampled shapes for the AKConv with different sizes. generating the inital sampled shapes for the AKConv with different sizes.
    # def _get_p_n(self, N, dtype):
    #     base_int = round(math.sqrt(self.num_param))
    #     row_number = self.num_param // base_int
    #     mod_number = self.num_param % base_int
    #     p_n_x, p_n_y = torch.meshgrid(
    #         torch.arange(0, row_number),
    #         torch.arange(0, base_int), indexing='xy')
    #     p_n_x = torch.flatten(p_n_x)
    #     p_n_y = torch.flatten(p_n_y)
    #     if mod_number > 0:
    #         mod_p_n_x, mod_p_n_y = torch.meshgrid(
    #             torch.arange(row_number, row_number + 1),
    #             torch.arange(0, mod_number), indexing='xy')
    #
    #         mod_p_n_x = torch.flatten(mod_p_n_x)
    #         mod_p_n_y = torch.flatten(mod_p_n_y)
    #         p_n_x, p_n_y = torch.cat((p_n_x, mod_p_n_x)), torch.cat((p_n_y, mod_p_n_y))
    #     p_n = torch.cat([p_n_x, p_n_y], 0)
    #     p_n = p_n.view(1, 2 * N, 1, 1, 1).type(dtype)
    #     return p_n
    def _get_p_n(self, N, dtype):
        base_int = round(self.num_param ** (1 / 3))  # 每个维度的基本长度
        x_count = base_int
        y_count = base_int
        z_count = self.num_param // (base_int ** 2)  # 确定完整的 Z 层数
        remaining = self.num_param % (base_int ** 2)  # 确定剩余的参数数目

        # 生成主网格
        p_n_x, p_n_y, p_n_z = torch.meshgrid(
            torch.arange(0, z_count),
            torch.arange(0, y_count),
            torch.arange(0, x_count),
            indexing='xy'
        )
        p_n_x = torch.flatten(p_n_x)
        p_n_y = torch.flatten(p_n_y)
        p_n_z = torch.flatten(p_n_z)

        # 如果有余数，需要生成额外的点
        if remaining > 0:
            extra_y_count = remaining // x_count  # 额外的完整 Y 层
            extra_x_count = remaining % x_count  # 剩下的 X 元素

            # Y 层的补充
            if extra_y_count > 0:
                mod_p_n_x1, mod_p_n_y1, mod_p_n_z1 = torch.meshgrid(
                    torch.arange(z_count, z_count + 1),
                    torch.arange(0, extra_y_count),
                    torch.arange(0, x_count),
                    indexing='xy'
                )
                mod_p_n_x1 = torch.flatten(mod_p_n_x1)
                mod_p_n_y1 = torch.flatten(mod_p_n_y1)
                mod_p_n_z1 = torch.flatten(mod_p_n_z1)
                p_n_x = torch.cat((p_n_x, mod_p_n_x1))
                p_n_y = torch.cat((p_n_y, mod_p_n_y1))
                p_n_z = torch.cat((p_n_z, mod_p_n_z1))

            # 最后的 X 补充
            if extra_x_count > 0:
                mod_p_n_x2, mod_p_n_y2, mod_p_n_z2 = torch.meshgrid(
                    torch.arange(z_count, z_count + 1),
                    torch.arange(extra_y_count, extra_y_count + 1),
                    torch.arange(0, extra_x_count),
                    indexing='xy'
                )
                mod_p_n_x2 = torch.flatten(mod_p_n_x2)
                mod_p_n_y2 = torch.flatten(mod_p_n_y2)
                mod_p_n_z2 = torch.flatten(mod_p_n_z2)
                p_n_x = torch.cat((p_n_x, mod_p_n_x2))
                p_n_y = torch.cat((p_n_y, mod_p_n_y2))
                p_n_z = torch.cat((p_n_z, mod_p_n_z2))

        # 合并 x, y, z 坐标
        p_n = torch.cat([p_n_x, p_n_y, p_n_z], 0)
        p_n = p_n.view(1, 3 * N, 1, 1, 1).type(dtype)  # N 表示需要的重复次数
        return p_n

    # no zero-padding
    def _get_p_0(self, h, w, d, N, dtype):
        p_0_x, p_0_y, p_0_z = torch.meshgrid(
            torch.arange(0, h * self.stride, self.stride),
            torch.arange(0, w * self.stride, self.stride),
            torch.arange(0, d * self.stride, self.stride), indexing='xy')

        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w, d).repeat(1, N, 1, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w, d).repeat(1, N, 1, 1, 1)
        p_0_z = torch.flatten(p_0_z).view(1, 1, h, w, d).repeat(1, N, 1, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y, p_0_z], 1).type(dtype)

        return p_0

    def _get_p(self, offset, dtype):
        N, h, w ,d = offset.size(1) // 3, offset.size(2), offset.size(3), offset.size(4)

        # (1, 3N, 1, 1, 1)
        p_n = self._get_p_n(N, dtype)
        # (1, 3N, h, w, d)
        p_0 = self._get_p_0(h, w, d, N, dtype)
        p = p_0 + p_n + offset
        return p

    def _get_x_q(self, x, q, N):
        """
        获取三维插值点对应的特征值。
        x: (b, c, h, w, d) 输入特征
        q: (b, h, w, d, 3N) 插值点坐标
        N: 单个维度的插值点数量
        """
        b, h, w, d, _ = q.size()
        padded_w = x.size(3)
        padded_d = x.size(4)
        c = x.size(1)

        # 将 x 展平为 (b, c, h*w*d)
        x = x.contiguous().view(b, c, -1)

        # 计算索引 (b, h, w, d, N)
        index = (
                q[..., :N] * (padded_w * padded_d) +  # offset_x * (w * d)
                q[..., N:2 * N] * padded_d +  # offset_y * d
                q[..., 2 * N:]  # offset_z
        )

        # 将索引扩展为 (b, c, h*w*d*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1, -1).contiguous().view(b, c, -1)

        # 限制索引范围，避免越界
        index = index.clamp(min=0, max=x.shape[-1] - 1)

        # 按索引提取特征值并 reshape
        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, d, N)

        return x_offset

    #  Stacking resampled features in the row direction.
    @staticmethod
    def _reshape_x_offset(x_offset, num_param):
        """
        将 x_offset 重塑为适合后续卷积操作的格式，支持 3D 输入。
        x_offset: 输入特征 (b, c, h, w, d, n)
        num_param: 参数数量，对应卷积核维度
        """

        # 获取形状信息
        b, c, h, w, d, n = x_offset.size()
        n3 = cube_root(n)
        # 重新排列张量，展开最后一个维度
        x = x_offset.view(b, c, h, w, d, n3, n3, n3)

        # 调整维度顺序，将子块并入主维度
        x = x.permute(0, 1, 2, 5, 3, 6, 4, 7)  # (56, 32, 24, 4, 24, 4, 24, 4)

        # 合并相邻维度，形成目标形状
        x = x.contiguous().view(b, c, h*n3, w*n3, d*n3)

        # 重新排列，将 (h, w, d, n) 展平为适合卷积的形式
        # x_offset = rearrange(x_offset, 'b c h w d')

        # 返回重塑后的张量
        return x


def cube_root(n):
    return round(math.pow(n, (1 / 3)))



if __name__ == '__main__':
    a = torch.ones(56, 32, 24, 24, 24)  #生成随机数
    b = AKConv(32,32,64)  #实例化
    c = b(a)
    print(c.size())

# if __name__ == "__main__":
#     input = torch.randint(
#         low=0,
#         high=255,
#         size=(56,4,96,96,96),
#         dtype=torch.float,
#     )
#     input = input.to("cuda:1")
#     segformer3D = SegFormer3D().to("cuda:1")
#     output = segformer3D(input)
#     print(output.shape)