[English version](https://github.com/bindog/pytorch-model-parallel/blob/master/README_EN.md)

# 显存均衡的模型并行实现(基于PyTorch、支持混合精度训练与分布式训练)

## 为什么要用模型并行？暴力数据并行不就好了？

在人脸和re-id领域，部分私有的数据集的label数量可达上百万/千万/亿的规模，此时fc层的参数量就足以把显卡容量撑满，导致只能使用较小的`batch_size`，训练速度较慢，效果不佳

## fc层模型并行我会，直接这样写不就好了？

```python
class FullyConnected(nn.Module):
    def __init__(self, in_dim, out_dim, num_gpu, model_parallel=False):
        super(FullyConnected, self).__init__()
        self.num_gpu = num_gpu
        self.model_parallel = model_parallel
        if model_parallel:
            self.fc_chunks = nn.ModuleList()
            for i in range(num_gpu):
                _class_num = out_dim // num_gpu
                if i < (out_dim % num_gpu):
                    _class_num += 1
                self.fc_chunks.append(
                    nn.Linear(in_dim, _class_num, bias=False).cuda(i)
                )
        else:
            self.classifier = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x):
        if self.model_parallel:
            x_list = []
            for i in range(self.num_gpu):
                _x = self.fc_chunks[i](x.cuda(i))
                x_list.append(_x)
            x = torch.cat(x_list, dim=1)
            return x
        else:
            return self.classifier(x)
```

类似的实现在这个基于PyTorch的[人脸项目](https://github.com/ZhaoJ9014/face.evoLVe.PyTorch/blob/d5e31893f7e30c0f82262e701463fd83d9725381/head/metrics.py#L41)也能够看到

这个方案能够部分解决这个问题，但是又会引入新的问题：显存占用不均衡。由于最后结果依然要concat回0号卡上，且loss计算依然在0号卡上，0号卡的显存占用以及计算负载显著高于其他卡。受制于此，依然无法使用较大的`batch_size`

## 这个项目解决了这些问题吗？

不仅解决了，还扩展到了更多场景下，支持人脸和re-id训练中常见的margin loss，支持混合精度训练与分布式训练。

**几点小小的优势：**

- 显存与计算负载合理分担到每张卡上，能够使用非常大的`batch_size`，训练得更加开心
- 只需做一些小小的修改就可以适配主流的margin loss，如`ArcFace`、`SphereFace`、`CosFace`、`AM-softmax`等等
- 相同的setting下对训练精度无影响（有数学推导保证结果正确）
- 在某些情况下甚至能加速训练（得益于优化后CrossEntropyLoss计算的过程中通信开销的降低）
- 支持混合精度训练与分布式训练

## 我该如何使用？

首先确认下你是否有必要使用模型并行：

- 数据集label规模是否在百万级以上？
- 模型的最后一层是否为fc层，是否使用CrossEntropyLoss？
- 显卡数量是否足够？（至少4~8张显卡）

如果以上答案均为肯定，那么你可以考虑使用模型并行。但是由于模型并行需要hack model和optimizer（分布式条件下更为复杂），目前需要自行移植到你的项目中。

- 普通的及混合精度训练可参考[master分支](https://github.com/bindog/pytorch-model-parallel/tree/master)
- 分布式训练可参考[dist分支](https://github.com/bindog/pytorch-model-parallel/tree/dist)，目前仍在完善中

## 其他框架怎么办？

原理都是相通的，其他框架如MXNet甚至有对分布式支持更为友好的`kvstore`可供使用

# 相关博客

- [http://bindog.github.io/blog/2019/09/05/gpu-memory-balanced-model-parallel/](http://bindog.github.io/blog/2019/09/05/gpu-memory-balanced-model-parallel/)

- [http://bindog.github.io/blog/2020/04/12/model-parallel-with-apex/](http://bindog.github.io/blog/2020/04/12/model-parallel-with-apex/)
