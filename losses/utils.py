import torch


def softmax_helper_dim0(x: torch.Tensor) -> torch.Tensor:
    """
    计算输入张量在第 0 维上的 softmax

    :param x: 输入张量
    :return: 经过 softmax 计算后的张量
    """
    return torch.softmax(x, 0)


def softmax_helper_dim1(x: torch.Tensor) -> torch.Tensor:
    """
    计算输入张量在第 1 维上的 softmax

    :param x: 输入张量
    :return: 经过 softmax 计算后的张量
    """
    return torch.softmax(x, 1)


def empty_cache(device: torch.device):
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    elif device.type == 'mps':
        from torch import mps
        mps.empty_cache()
    else:
        pass


class dummy_context(object):
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
