import torch
import torch.nn as nn
import torch.distributed as dist
from torch.autograd import Function


class AllGatherFunc(Function):
    @staticmethod
    def forward(ctx, tensor, *gather_list):
        gather_list = list(gather_list)
        dist.all_gather(gather_list, tensor)
        return tuple(gather_list)

    @staticmethod
    def backward(ctx, *grads):
        grad_list = list(grads)
        grad_out = grad_list[dist.get_rank()]
        return (grad_out, *[None for _ in range(len(grad_list))])

AllGather = AllGatherFunc.apply
