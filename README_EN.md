# A memory balanced and communication efficient FullyConnected layer model parallel implementation in PyTorch

## Why we need model parallel? Why not use the DataParallel?

Well, in face and re-id (person re-identification) areas, the number of labels in some private datasets may exceeds 1 million/10 millions/100 millions, and the parameters of the fully connected layer will occupy the whole GPU memory, and we can only use a small batch size which will result in slow training speed and poor evaluation performance

##  Fully connected layer with model parallel? It's simple!

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
Similar implementation can also be found [here](https://github.com/ZhaoJ9014/face.evoLVe.PyTorch/blob/d5e31893f7e30c0f82262e701463fd83d9725381/head/metrics.py#L41)

this implementation can only solve part of the problem, and it will introduce new issues, GPU memory imbalanced usage between different gpus. Cause all the results will concat at GPU 0, and the loss calculation also happends at GPU 0, the GPU memory usage and computaion load will be much higher in GPU 0 compare to other GPUs, we still can not use big batch size.

## Did this repository solve the problem?

Yes, and it extends to more occasions, like margin loss, mixed precison training and distributed training

some advantages:

- GPU memory usage and computation load will balanced among all GPUs, we can use a big batch size, life will be easier:-)
- support most of the margin loss in face and re-id areas, like `ArcFace`, `SphereFace`, `CosFace`, `AM-softmax` and so on
- it won't affect your evaluation result after training with the model parallel
- sometimes speed up your training (due to lower communication cost in optimized CrossEntropyLoss)
- support mixed precision training and distributed training

## How can I use this?

First make sure you do need model parallel:

- Is the number of labels in your datasets exceed 1 million?
- If the last layer of your model is fully connected layer? And Did you use the CrossEntropyLoss?
- Do you have enough GPUs? (at least 4~8 GPUs)

If the anwser of all the above questions is yes, and you can consider using the model parallel. But as it requires to hack into the model and optimizer, you will need to migrate this to your repository by yourself

- normal training and mixed precison training, refer to [master branch](https://github.com/bindog/pytorch-model-parallel/tree/master)
- distributed training, refer to [dist branch](https://github.com/bindog/pytorch-model-parallel/tree/dist)

## what about other deep learning frameworks?

the principle is the same, other frameworks like MXNet has a better support (kvstore) for distributed training

# Chinese blogs

- [http://bindog.github.io/blog/2019/09/05/gpu-memory-balanced-model-parallel/](http://bindog.github.io/blog/2019/09/05/gpu-memory-balanced-model-parallel/)
- [http://bindog.github.io/blog/2020/04/12/model-parallel-with-apex/](http://bindog.github.io/blog/2020/04/12/model-parallel-with-apex/)
