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
        # args[1] --> fp16 flag (type: Tensor)
        # args[2:num_splits + 2] --> one-hot label parts (type: Tensor or SparseTensor)
        # args[num_splits + 2:] --> fc logit parts
        ctx.num_splits = (len(args) - 2) // 2
        ctx.compute_loss = args[0]
        ctx.fp16 = args[1]
        ctx.batch_size = args[2].size()[0]
        ctx.label_split = args[2:ctx.num_splits + 2]
        # for numerical stability
        max_list = []
        for arg in args[ctx.num_splits + 2:]:
            m, _ = torch.max(arg, dim=1, keepdim=True)
            max_list.append(m)
        mc = torch.cat(max_list, dim=1)
        m, _ = torch.max(mc, dim=1, keepdim=True)
        nargs = [arg - m.to(gpu_id) for gpu_id, arg in enumerate(args[ctx.num_splits + 2:])]

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

        loss = torch.ones(1)
        if ctx.compute_loss:
            _loss_list = []
            for gpu_id, softmax in enumerate(softmax_list):
                idx = ctx.label_split[gpu_id]._indices()
                _loss = torch.zeros(ctx.batch_size).to(gpu_id)
                _loss.scatter_(dim=0, index=idx[0], src=softmax[tuple(idx)])
                _loss_list.append(_loss)
            _loss = comm.reduce_add(_loss_list, destination=0)
            log_loss = -torch.log(_loss)
            loss = torch.mean(log_loss)

        return loss

    @staticmethod
    @once_differentiable
    @amp.float_function
    def backward(ctx, loss_grad):
        grad_logit_list = []
        for gpu_id, softmax in enumerate(ctx.saved_variables):
            grad_logit = (softmax - ctx.label_split[gpu_id].float()) / ctx.batch_size
            # scaled loss
            grad_logit_list.append(grad_logit * loss_grad.item())
        if ctx.fp16:
            grad_logit_list = [g.half() for g in grad_logit_list]
        grad_logit_list = [None]*(ctx.num_splits + 2) + grad_logit_list
        return tuple(grad_logit_list)

MPCrossEntropy = ModelParallelCrossEntropyFunc.apply

class ModelParallelCrossEntropy(nn.Module):
    def __init__(self):
        super(ModelParallelCrossEntropy, self).__init__()

    # args[0] --> compute loss flag (type: Bool)
    # args[1] --> fp16 flag (type: Bool)
    # args[2] --> one-hot label parts (type: Tuple)
    # args[3:] --> fc logit parts
    def forward(self, *args):
        # The new staticmethod style requires type of all input args to be Tenosr
        # so we need to convert the args here
        compute_loss = torch.ones(1) if args[0] else torch.zeros(1)
        fp16 = torch.ones(1) if args[1] else torch.zeros(1)
        new_args = (compute_loss, fp16, *args[2], *args[3:])
        return MPCrossEntropy(*new_args)
