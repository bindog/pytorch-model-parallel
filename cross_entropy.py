import torch
import torch.nn as nn
import torch.cuda.comm as comm
from torch.autograd import Function

from torch.autograd.function import once_differentiable
from apex import amp


# The new staticmethod style combine with apex
# Learn from https://github.com/lbin/DCNv2/blob/master/dcn_v2.py


class ModelParallelCrossEntropyFunc(Function):

    @staticmethod
    @amp.float_function
    def forward(ctx, *args):
        # args[0] --> compute loss flag (type: Tensor)
        # args[1:num_splits + 1] --> one-hot label parts
        # args[num_splits + 1:] --> fc logit parts
        ctx.num_splits = (len(args) - 1) // 2
        ctx.compute_loss = args[0]
        ctx.batch_size = args[1].size()[0]
        ctx.label_split = args[1:ctx.num_splits + 1]
        # for numerical stability
        max_list = []
        for arg in args[ctx.num_splits + 1:]:
            m, _ = torch.max(arg, dim=1, keepdim=True)
            max_list.append(m)
        mc = torch.cat(max_list, dim=1)
        m, _ = torch.max(mc, dim=1, keepdim=True)
        nargs = [arg - m.to(gpu_id) for gpu_id, arg in enumerate(args[ctx.num_splits + 1:])]

        # get exp sum
        exp_logit_list = []
        exp_sum_list = []
        for gpu_id, narg in enumerate(nargs):
            exp_logit = torch.exp(narg)
            exp_logit_list.append(exp_logit)
            exp_sum = torch.sum(exp_logit, dim=1, keepdim=True)
            exp_sum_list.append(exp_sum)
        exp_sum_all = comm.reduce_add(exp_sum_list, 0)

        # compute softmax output
        softmax_list = []
        for gpu_id, narg in enumerate(nargs):
            softmax = exp_logit_list[gpu_id] / exp_sum_all.to(gpu_id)
            softmax_list.append(softmax)
        ctx.save_for_backward(*softmax_list)

        loss = torch.zeros(1)
        if ctx.compute_loss:
            _loss_list = []
            for gpu_id, softmax in enumerate(softmax_list):
                _loss = torch.sum(softmax * ctx.label_split[gpu_id], dim=1)
                _loss_list.append(_loss)
            _loss = comm.reduce_add(_loss_list, 0)
            log_loss = -torch.log(_loss)
            loss = torch.mean(log_loss)

        return loss

    @staticmethod
    @once_differentiable
    @amp.float_function
    def backward(ctx, loss_grad):
        grad_logit_list = []
        for gpu_id, softmax in enumerate(ctx.saved_variables):
            grad_logit = (softmax - ctx.label_split[gpu_id]) / ctx.batch_size
            # grad_logit_list.append(grad_logit)
            grad_logit_list.append(grad_logit.half())
        # print("="*70)
        # print("debug fc grad...")
        # print(grad_logit_list[0])
        # print("="*70)
        grad_logit_list = [None]*(ctx.num_splits + 1) + grad_logit_list
        return tuple(grad_logit_list)

MPCrossEntropy = ModelParallelCrossEntropyFunc.apply

class ModelParallelCrossEntropy(nn.Module):
    def __init__(self):
        super(ModelParallelCrossEntropy, self).__init__()

    # args[0] --> compute loss flag (type: Bool)
    # args[1] --> one-hot label parts (type: Tuple)
    # args[2:] --> fc logit parts
    def forward(self, *args):
        # The new staticmethod style requires type of all input args to be Tenosr
        # so we need to convert the args here
        compute_loss = torch.ones(1) if args[0] else torch.zeros(1)
        new_args = (compute_loss, *args[1], *args[2:])
        return MPCrossEntropy(*new_args)
