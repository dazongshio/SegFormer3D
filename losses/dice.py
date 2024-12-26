import torch
from torch import nn
from torch import distributed
from typing import Any, Optional, Tuple, Callable
from losses.utils import softmax_helper_dim1

class SoftDiceLoss(nn.Module):
    def __init__(self, apply_nonlin: Callable = None, batch_dice: bool = False, do_bg: bool = True, smooth: float = 1.,
                 ddp: bool = True, clip_tp: float = None):
        """
        初始化 SoftDiceLoss 类

        :param apply_nonlin: 应用于预测的非线性函数（如 sigmoid 或 softmax）
        :param batch_dice: 是否在 batch 维度上计算 Dice 系数
        :param do_bg: 是否包含背景类的计算
        :param smooth: 平滑项，用于防止分母为零
        :param ddp: 是否在分布式数据并行 (DDP) 中使用
        :param clip_tp: 限制 true positive (TP) 的最小值
        """
        super(SoftDiceLoss, self).__init__()
        self.do_bg = do_bg  # 是否包含背景类
        self.batch_dice = batch_dice  # 是否在 batch 维度上计算
        self.apply_nonlin = apply_nonlin  # 非线性函数
        self.smooth = smooth  # 平滑项
        self.clip_tp = clip_tp  # 限制 TP 的最小值
        self.ddp = ddp  # 是否使用 DDP

    def forward(self, x, y, loss_mask=None):
        """
        前向传播，计算 Soft Dice Loss

        :param x: 模型预测结果
        :param y: 目标标签
        :param loss_mask: 可选的损失掩码
        :return: 计算得到的 Soft Dice Loss
        """
        shp_x = x.shape  # 获取预测张量的形状

        # 根据 batch_dice 参数确定计算 Dice 的维度
        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))  # 包括 batch 维度
        else:
            axes = list(range(2, len(shp_x)))  # 不包括 batch 维度

        # 如果提供了非线性函数，则对预测结果应用该函数
        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        # 计算 TP（真阳性）、FP（假阳性）、FN（假阴性）和 TN（真阴性）
        tp, fp, fn, _ = get_tp_fp_fn_tn(x, y, axes, loss_mask, False)

        # 如果使用 DDP 并且需要在 batch 维度计算，则汇总所有进程的 TP、FP 和 FN
        if self.ddp and self.batch_dice:
            tp = AllGatherGrad.apply(tp).sum(0)  # 汇总 TP
            fp = AllGatherGrad.apply(fp).sum(0)  # 汇总 FP
            fn = AllGatherGrad.apply(fn).sum(0)  # 汇总 FN

        # 如果设置了 clip_tp，则将 TP 限制在最小值 clip_tp
        if self.clip_tp is not None:
            tp = torch.clip(tp, min=self.clip_tp, max=None)

        # 计算 Dice 系数的分子和分母
        nominator = 2 * tp  # 分子：2 * TP
        denominator = 2 * tp + fp + fn  # 分母：2 * TP + FP + FN

        # 计算 Soft Dice 系数
        dc = (nominator + self.smooth) / (torch.clip(denominator + self.smooth, 1e-8))

        # 如果不计算背景类，则去除背景类的结果
        if not self.do_bg:
            if self.batch_dice:
                # dc = dc[1:]  # 去除 batch 维度上的背景类
                dc = dc.as_tensor()[1:]  # 将 dc 转换为普通 Tensor 后再切片
            else:
                # dc = dc[:, 1:]  # 去除通道维度上的背景类
                dc = dc.as_tensor()[:, 1:]  # 将 dc 转换为普通 Tensor 后再切片

        # 返回负的平均 Dice 系数作为损失值
        return -dc.mean()



class MemoryEfficientSoftDiceLoss(nn.Module):
    def __init__(self, apply_nonlin: Callable = None, batch_dice: bool = False, do_bg: bool = True, smooth: float = 1.,
                 ddp: bool = True):
        """
        saves 1.6 GB on Dataset017 3d_lowres
        """
        super(MemoryEfficientSoftDiceLoss, self).__init__()

        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.ddp = ddp

    def forward(self, x, y, loss_mask=None):
        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        # make everything shape (b, c)
        axes = tuple(range(2, x.ndim))

        with torch.no_grad():
            if x.ndim != y.ndim:
                y = y.view((y.shape[0], 1, *y.shape[1:]))

            if x.shape == y.shape:
                # if this is the case then gt is probably already a one hot encoding
                y_onehot = y
            else:
                y_onehot = torch.zeros(x.shape, device=x.device, dtype=torch.bool)
                y_onehot.scatter_(1, y.long(), 1)

            if not self.do_bg:
                y_onehot = y_onehot[:, 1:]

            sum_gt = y_onehot.sum(axes) if loss_mask is None else (y_onehot * loss_mask).sum(axes)

        # this one MUST be outside the with torch.no_grad(): context. Otherwise no gradients for you
        if not self.do_bg:
            x = x[:, 1:]

        if loss_mask is None:
            intersect = (x * y_onehot).sum(axes)
            sum_pred = x.sum(axes)
        else:
            intersect = (x * y_onehot * loss_mask).sum(axes)
            sum_pred = (x * loss_mask).sum(axes)

        if self.batch_dice:
            if self.ddp:
                intersect = AllGatherGrad.apply(intersect).sum(0)
                sum_pred = AllGatherGrad.apply(sum_pred).sum(0)
                sum_gt = AllGatherGrad.apply(sum_gt).sum(0)

            intersect = intersect.sum(0)
            sum_pred = sum_pred.sum(0)
            sum_gt = sum_gt.sum(0)

        dc = (2 * intersect + self.smooth) / (torch.clip(sum_gt + sum_pred + self.smooth, 1e-8))

        dc = dc.mean()
        return -dc


class SoftSkeletonRecallLoss(nn.Module):
    def __init__(self, apply_nonlin: Callable = None, batch_dice: bool = False, do_bg: bool = True, smooth: float = 1.,
                 ddp: bool = True):
        """
        saves 1.6 GB on Dataset017 3d_lowres
        """
        super(SoftSkeletonRecallLoss, self).__init__()

        if do_bg:
            raise RuntimeError("skeleton recall does not work with background")
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.ddp = ddp

    def forward(self, x, y, loss_mask=None):
        shp_x, shp_y = x.shape, y.shape

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        x = x[:, 1:]

        # make everything shape (b, c)
        axes = list(range(2, len(shp_x)))

        with torch.no_grad():
            if len(shp_x) != len(shp_y):
                y = y.view((shp_y[0], 1, *shp_y[1:]))

            if all([i == j for i, j in zip(shp_x, shp_y)]):
                # if this is the case then gt is probably already a one hot encoding
                y_onehot = y[:, 1:]
            else:
                gt = y.long()
                y_onehot = torch.zeros(shp_x, device=x.device, dtype=y.dtype)
                y_onehot.scatter_(1, gt, 1)
                y_onehot = y_onehot[:, 1:]

            sum_gt = y_onehot.sum(axes) if loss_mask is None else (y_onehot * loss_mask).sum(axes)

        inter_rec = (x * y_onehot).sum(axes) if loss_mask is None else (x * y_onehot * loss_mask).sum(axes)

        if self.ddp and self.batch_dice:
            inter_rec = AllGatherGrad.apply(inter_rec).sum(0)
            sum_gt = AllGatherGrad.apply(sum_gt).sum(0)

        if self.batch_dice:
            inter_rec = inter_rec.sum(0)
            sum_gt = sum_gt.sum(0)

        rec = (inter_rec + self.smooth) / (torch.clip(sum_gt + self.smooth, 1e-8))

        rec = rec.mean()
        return -rec


def get_tp_fp_fn_tn(net_output, gt, axes=None, mask=None, square=False):
    """
    计算真阳性 (TP)、假阳性 (FP)、假阴性 (FN) 和真阴性 (TN)

    :param net_output: 模型的输出结果，形状为 (b, c, x, y(, z))
    :param gt: 真实标签，形状可以是 (b, 1, x, y(, z)) 或 (b, x, y(, z)) 或 one-hot 编码 (b, c, x, y(, z))
    :param axes: 指定在哪些维度上求和，例如 (1, 2, 3)
    :param mask: 可选的掩码张量，形状为 (b, 1, x, y(, z))，用于指定有效区域
    :param square: 如果为 True，则在求和前对 TP、FP 和 FN 进行平方
    :return: TP, FP, FN, TN 四个张量
    """
    if axes is None:
        axes = tuple(range(2, net_output.ndim))  # 默认从第 2 维到最后一维求和

    with torch.no_grad():
        # if net_output.ndim != gt.ndim:
        #     gt = gt.view((gt.shape[0], 1, *gt.shape[1:]))  # 将 gt 的形状调整为与 net_output 一致

        if net_output.shape == gt.shape:
            # 如果 net_output 和 gt 的形状一致，说明 gt 可能已经是 one-hot 编码
            y_onehot = gt.bool()
        else:
            # 将 gt 转换为 one-hot 编码
            y_onehot = torch.zeros(net_output.shape, device=net_output.device, dtype=torch.bool)
            y_onehot.scatter_(1, gt.long(), 1)

    tp = net_output * y_onehot  # 真阳性
    fp = net_output * (~y_onehot)  # 假阳性
    fn = (1 - net_output) * y_onehot  # 假阴性
    tn = (1 - net_output) * (~y_onehot)  # 真阴性

    if mask is not None:
        with torch.no_grad():
            # 将掩码扩展到与 TP、FP、FN、TN 一致的形状
            mask_here = torch.tile(mask, (1, tp.shape[1], *[1 for _ in range(2, tp.ndim)]))
        tp *= mask_here
        fp *= mask_here
        fn *= mask_here
        tn *= mask_here

    if square:
        tp = tp ** 2
        fp = fp ** 2
        fn = fn ** 2
        tn = tn ** 2

    if len(axes) > 0:
        # 在指定维度上求和
        tp = tp.sum(dim=axes, keepdim=False)
        fp = fp.sum(dim=axes, keepdim=False)
        fn = fn.sum(dim=axes, keepdim=False)
        tn = tn.sum(dim=axes, keepdim=False)

    return tp, fp, fn, tn


def print_if_rank0(*args):
    if distributed.get_rank() == 0:
        print(*args)


class AllGatherGrad(torch.autograd.Function):
    # stolen from pytorch lightning
    @staticmethod
    def forward(
            ctx: Any,
            tensor: torch.Tensor,
            group: Optional["torch.distributed.ProcessGroup"] = None,
    ) -> torch.Tensor:
        ctx.group = group

        gathered_tensor = [torch.zeros_like(tensor) for _ in range(torch.distributed.get_world_size())]

        torch.distributed.all_gather(gathered_tensor, tensor, group=group)
        gathered_tensor = torch.stack(gathered_tensor, dim=0)

        return gathered_tensor

    @staticmethod
    def backward(ctx: Any, *grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        grad_output = torch.cat(grad_output)

        torch.distributed.all_reduce(grad_output, op=torch.distributed.ReduceOp.SUM, async_op=False, group=ctx.group)

        return grad_output[torch.distributed.get_rank()], None


if __name__ == '__main__':
    pred = torch.rand((4, 8, 32, 32, 32))
    ref = torch.randint(0, 8, (4,8, 32, 32, 32))

    dl_old = SoftDiceLoss(apply_nonlin=softmax_helper_dim1, batch_dice=True, do_bg=False, smooth=0, ddp=False)
    dl_new = MemoryEfficientSoftDiceLoss(apply_nonlin=softmax_helper_dim1, batch_dice=True, do_bg=False, smooth=0,
                                         ddp=False)
    res_old = dl_old(pred, ref)
    res_new = dl_new(pred, ref)
    print(res_old, res_new)
