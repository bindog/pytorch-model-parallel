import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed import ReduceOp
from torch.autograd import Function


class AllGatherFunc(Function):
    @staticmethod
    def forward(ctx, tensor, *gather_list):
        # ctx.save_for_backward(tensor)
        gather_list = list(gather_list)
        dist.all_gather(gather_list, tensor)
        return tuple(gather_list)

    @staticmethod
    def backward(ctx, *grads):
        # func_input = ctx.saved_tensors
        # grad_out = torch.zeros_like(func_input)
        grad_list = list(*grads)
        grad_out = grad_list[dist.get_rank()]
        return (grad_out, *grad_list)
    # @staticmethod
    # def forward(ctx, tensor, group, inplace, *gather_list):
    #     ctx.save_for_backward(tensor)
    #     ctx.group = group
    #     gather_list = list(gather_list)
    #     if not inplace:
    #         gather_list = [torch.zeros_like(g) for g in gather_list]
    #     dist.all_gather(gather_list, tensor, group)
    #     return tuple(gather_list)

    # @staticmethod
    # def backward(ctx, *grads):
    #     input, = ctx.saved_tensors
    #     grad_out = torch.zeros_like(input)
    #     # TODO fix
    #     # https://github.com/ag14774/diffdist/blob/master/diffdist/extra_collectives.py
    #     dist_extra.reduce_scatter(grad_out, list(grads), group=ctx.group)
    #     return (grad_out, None, None) + grads

AllGather = AllGatherFunc.apply

# class AsyncOpList(object):
#     def __init__(self, ops):
#         self.ops = ops

#     def wait(self):
#         for op in self.ops:
#             op.wait()

#     def is_completed(self):
#         for op in self.ops:
#             if not op.is_completed():
#                 return False
#         return True


# def reduce_scatter(tensor,
#                    tensor_list,
#                    op=ReduceOp.SUM,
#                    group=dist.group.WORLD,
#                    async_op=False):
#     rank = dist.get_rank(group)
#     if tensor is None:
#         tensor = tensor_list[rank]
#     if tensor.dim() == 0:
#         tensor = tensor.view(-1)
#     tensor[:] = tensor_list[rank]
#     ops = []
#     for i in range(dist.get_world_size(group)):
#         if i == rank:
#             tmp = dist.reduce(tensor, rank, op, group, async_op=True)
#         else:
#             tmp = dist.reduce(tensor_list[i], i, op, group, async_op=True)
#         ops.append(tmp)

#     oplist = AsyncOpList(ops)
#     if async_op:
#         return oplist
#     else:
#         oplist.wait()
