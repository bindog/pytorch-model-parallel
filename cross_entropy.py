import torch
import torch.nn as nn
import torch.cuda.comm as comm
from torch.autograd import Function

from torch.autograd.function import once_differentiable
from apex import amp


class ModelParallelCrossEntropyFunc(Function):

    @staticmethod
    @amp.float_function
    def forward(ctx, *args):  # args is list of logit parts
        ctx.num_splits = (len(args) - 1) // 2
        ctx.compute_loss = args[0]
        ctx.batch_size = args[1].size()[0]
        ctx.label_split = args[1: 1 + ctx.num_splits]
        # for numerical stability
        max_list = []
        for arg in args[1 + ctx.num_splits:]:
            m, _ = torch.max(arg, dim=1, keepdim=True)
            max_list.append(m)
        mc = torch.cat(max_list, dim=1)
        m, _ = torch.max(mc, dim=1, keepdim=True)
        nargs = [arg - m.to(gpu_id) for gpu_id, arg in enumerate(args[1 + ctx.num_splits:])]

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
        grad_logit_list = [None]*(1 + ctx.num_splits) + grad_logit_list
        return tuple(grad_logit_list)

MPCrossEntropy = ModelParallelCrossEntropyFunc.apply

class ModelParallelCrossEntropy(nn.Module):
    def __init__(self):
        super(ModelParallelCrossEntropy, self).__init__()

    # args[0] is compute loss flag, args[1] is label_tuple
    # args[2:] is logit parts
    def forward(self, *args):
        # return ModelParallelCrossEntropyFunc(args[0], args[1])(*args[2:])
        new_args = []
        # compute_loss
        if args[0]:
            new_args.append(torch.ones(1))
        else:
            new_args.append(torch.zeros(1))
        # label_tuple
        for label_t in args[1]:
            new_args.append(label_t)
        for logit_t in args[2:]:
            new_args.append(logit_t)
        return MPCrossEntropy(*new_args)
