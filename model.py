import torch
import torch.nn as nn
from torch.autograd import Variable

from resnet import resnet50


class ft_net(nn.Module):
    def __init__(self, feature_dim, num_classes, num_gpus=1, am=False, model_parallel=False, class_split=None):
        super(ft_net, self).__init__()
        self.backbone_and_feature = resnet50(pretrained=True, feature_dim=feature_dim)
        if am:
            self.classifier = FullyConnected_AM(feature_dim, num_classes, num_gpus, model_parallel, class_split)
        else:
            self.classifier = FullyConnected(feature_dim, num_classes, num_gpus, model_parallel, class_split)

    def forward(self, x, labels=None):
        x = self.backbone_and_feature(x)
        x = self.classifier(x, labels)
        return x


class ft_net_dist(nn.Module):
    def __init__(self, feature_dim, num_classes, num_gpus=1, am=False, model_parallel=False, class_split=None):
        super(ft_net_dist, self).__init__()
        self.backbone_and_feature = resnet50(pretrained=True, feature_dim=feature_dim)

    def forward(self, x, labels=None):
        x = self.backbone_and_feature(x)
        return x


class FullyConnected(nn.Module):
    def __init__(self, in_dim, out_dim, num_gpus=1, model_parallel=False, class_split=None):
        super(FullyConnected, self).__init__()
        self.num_gpus = num_gpus
        self.model_parallel = model_parallel
        if model_parallel:
            self.fc_chunks = nn.ModuleList()
            for i in range(num_gpus):
                self.fc_chunks.append(
                    nn.Linear(in_dim, class_split[i], bias=False).cuda(i)
                )
        else:
            self.fc = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x, labels=None):
        if self.model_parallel:
            x_list = []
            for i in range(self.num_gpus):
                _x = self.fc_chunks[i](x.cuda(i))
                x_list.append(_x)
            return tuple(x_list)
        else:
            return self.fc(x)


class FullyConnected_AM(nn.Module):
    def __init__(self, in_dim, out_dim, num_gpus=1, model_parallel=False, class_split=None, margin=0.35, scale=30):
        super(FullyConnected_AM, self).__init__()
        self.num_gpus = num_gpus
        self.model_parallel = model_parallel
        if self.model_parallel:
            self.am_branches = nn.ModuleList()
            for i in range(num_gpus):
                self.am_branches.append(AM_Branch(in_dim, class_split[i], margin, scale).cuda(i))
        else:
            self.am = AM_Branch(in_dim, out_dim, margin, scale)

    def forward(self, x, labels=None):
        if self.model_parallel:
            output_list = []
            for i in range(self.num_gpus):
                output = self.am_branches[i](x.cuda(i), labels[i])
                output_list.append(output)
            return tuple(output_list)
        else:
            return self.am(x, labels)


class AM_Branch(nn.Module):
    def __init__(self, in_dim, out_dim, margin=0.35, scale=30):
        super(AM_Branch, self).__init__()
        self.m = margin
        self.s = scale
        #  training parameter
        self.weight = nn.Parameter(torch.Tensor(in_dim, out_dim), requires_grad=True)
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x, label):
        x_norm = x.pow(2).sum(1).pow(0.5)
        w_norm = self.weight.pow(2).sum(0).pow(0.5)
        cos_theta = torch.mm(x, self.weight) / x_norm.view(-1, 1) / w_norm.view(1, -1)
        cos_theta = cos_theta.clamp(-1, 1)
        phi = cos_theta - self.m

        index = label.data
        index = index.byte()

        output = cos_theta * 1.0
        output[index] = phi[index]
        output *= self.s

        return output


if __name__ == '__main__':
    net = ft_net(256, 65536, 4, False)
    print(net)
    input = Variable(torch.FloatTensor(8, 3, 256, 128))
    output = net(input)
    print('net output size:')
    if isinstance(output, tuple):
        for o in output:
            print(o.shape)
    else:
        print(output.shape)
