import torch
import torch.nn as nn
import torch.cuda.comm as comm
from torch.autograd import Function


class ModelParallelCrossEntropy(nn.Module):
    def __init__(self):
        super(ModelParallelCrossEntropy, self).__init__()

    # args[0] is compute loss flag, args[1] is label_tuple
    # args[2:] is logit parts
    def forward(self, *args):
        return ModelParallelCrossEntropyFunc(args[0], args[1])(*args[2:])


class ModelParallelCrossEntropyFunc(Function):
    def __init__(self, compute_loss, label_tuple):
        self.batch_size = label_tuple[0].size()[0]
        self.compute_loss = compute_loss
        self.label_split = label_tuple

    def forward(self, *args):  # args is list of logit parts
        # for numerical stability
        max_list = []
        for arg in args:
            m, _ = torch.max(arg, dim=1, keepdim=True)
            max_list.append(m)
        mc = torch.cat(max_list, dim=1)
        m, _ = torch.max(mc, dim=1, keepdim=True)
        nargs = [arg - m.to(gpu_id) for gpu_id, arg in enumerate(args)]

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
        self.save_for_backward(*softmax_list)

        loss = torch.zeros(1)
        if self.compute_loss:
            _loss_list = []
            for gpu_id, softmax in enumerate(softmax_list):
                _loss = torch.sum(softmax * self.label_split[gpu_id], dim=1)
                _loss_list.append(_loss)
            _loss = comm.reduce_add(_loss_list, 0)
            log_loss = -torch.log(_loss)
            loss = torch.mean(log_loss)

        return loss

    def backward(self, loss_grad):
        grad_logit_list = []
        for gpu_id, softmax in enumerate(self.saved_variables):
            grad_logit = (softmax - self.label_split[gpu_id]) / self.batch_size
            # grad_logit_list.append(grad_logit)
            grad_logit_list.append(grad_logit.half())
        return tuple(grad_logit_list)
