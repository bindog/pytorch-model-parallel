import os
import time
import logging
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms
from torch.utils.data.distributed import DistributedSampler

from model import ft_net_dist
from cross_entropy import DistModelParallelCrossEntropy
from utils import *

try:
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp, optimizers
    from apex.multi_tensor_apply import multi_tensor_applier
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")


def get_data_loader(data_path, batch_size):
    transform_train_list = [
        transforms.Resize((384, 192), interpolation=3),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
    data_transforms = transforms.Compose(transform_train_list)
    image_dataset = datasets.ImageFolder(data_path, data_transforms)
    sampler = DistributedSampler(image_dataset);
    dataloader = torch.utils.data.DataLoader(
                                    image_dataset,
                                    batch_size=batch_size,
                                    shuffle=False,
                                    num_workers=8,
                                    pin_memory=True,
                                    sampler=sampler
                                    )
    return len(image_dataset.classes), dataloader, sampler


def train_model(opt, data_loader, sampler, model, part_fc, criterion, optimizer, optimizer_part_fc, class_split):
# def train_model(opt, data_loader, sampler, model, criterion, optimizer, class_split):
    if opt.rank == 0:
        logging.info("Start training...")
    for epoch in range(opt.num_epochs):
        if opt.distributed:
            sampler.set_epoch(epoch)
        data_loader_iter = iter(data_loader)
        for step in range(len(data_loader)):
            start_time = time.time()
            images, labels = data_loader_iter.next()
            images = images.cuda()
            labels = labels.cuda()
            rank_batch_size = labels.size(0)

            total_labels, onehot_labels = get_sparse_onehot_label_dist(opt, labels, class_split)
            onehot_label = onehot_labels[opt.rank]

            # Forward
            optimizer.zero_grad()
            optimizer_part_fc.zero_grad()

            # collect all features
            features = model(images)
            features_gather = [torch.zeros_like(features) for _ in range(opt.world_size)]
            dist.all_gather(features_gather, features)
            all_features = torch.cat(features_gather, dim=0)
            logit = part_fc(all_features.cuda())
            # # get logit
            # logit = model(images)

            # Loss calculation
            compute_loss = step > 0 and step % 10 == 0
            loss = criterion(logit, onehot_label, compute_loss, opt.fp16, opt.world_size)
            # loss = criterion(logit, labels)
            if opt.rank == 0 and step > 0 and step % 10 == 0:
                getBack(loss)
                exit()

            # Backward
            scale = 1.0
            with amp.scale_loss(loss, [optimizer, optimizer_part_fc]) as scaled_loss:
                scale = scaled_loss.item() / loss.item()  # for debug purpose
                scaled_loss.backward()
            if opt.rank == 0 and step > 0 and step % 10 == 0:
                print("debug fc gradient", opt.rank, part_fc.weight.grad[0][:20])
                # print("debug cnn gradient", opt.rank, model.module.backbone_and_feature.conv1.weight.grad[0][0][0])
                print("debug cnn gradient", opt.rank, model.module.backbone_and_feature.conv1.weight.grad)
            optimizer.step()
            optimizer_part_fc.step()
            # Log training progress
            total_batch_size = rank_batch_size * opt.world_size
            if step > 0 and step % 10 == 0:
                example_per_second = total_batch_size / float(time.time() - start_time)

                batch_acc = compute_batch_acc_dist(opt, logit, total_labels, total_batch_size, class_split)
                if opt.rank == 0:
                    logging.info(
                            "epoch [%.3d] iter = %d loss = %.3f scale = %.3f acc = %.4f example/sec = %.2f" %
                            (epoch+1, step, loss.item(), scale, batch_acc, example_per_second)
                        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s %(filename)s] %(message)s")

    parser = argparse.ArgumentParser(description='Training')
    # parser.add_argument('--gpus', default='0,1,2,3', type=str, help='0,1,2,3')
    parser.add_argument('--data_path', default='/your/data/path/Market-1501/train', type=str, help='training data path')
    parser.add_argument('--num_epochs', default=15, type=int, help='number of epochs')
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    parser.add_argument('--num_classes', default=0, type=int, help='number of classes')
    parser.add_argument('--lr', default=0.05, type=float, help='learning rate')
    parser.add_argument('--am', action="store_true", help='use am-softmax')
    parser.add_argument('--model_parallel', action="store_true", help='use model parallel')
    parser.add_argument('--fp16', action="store_true", help='use mixed-precision')
    parser.add_argument("--local_rank", default=0, type=int)
    opt = parser.parse_args()

    # os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
    cudnn.benchmark = True
    # gpu_ids = opt.gpus.split(",")
    # num_gpus = len(gpu_ids)

    opt.distributed = True
    # if 'WORLD_SIZE' in os.environ:
    #     opt.distributed = int(os.environ['WORLD_SIZE']) > 1

    opt.gpu = 0
    opt.world_size = 1

    if opt.distributed:
        opt.gpu = opt.local_rank
        torch.cuda.set_device(opt.gpu)
        torch.distributed.init_process_group(backend='nccl',
                                             init_method='env://')
        opt.world_size = torch.distributed.get_world_size()
        opt.rank = dist.get_rank()

    num_classes, data_loader, sampler = get_data_loader(opt.data_path, opt.batch_size // opt.world_size)
    if opt.num_classes < num_classes:
        opt.num_classes = num_classes

    class_split = None
    if opt.model_parallel:
        # If using model parallel, split the number of classes
        # accroding to the number of GPUs
        class_split = get_class_split(opt.num_classes, opt.world_size)

    model = ft_net_dist(
            feature_dim=256,
            num_classes=opt.num_classes,
            num_gpus=opt.world_size,
            am=opt.am,
            model_parallel=opt.model_parallel,
            class_split=class_split
        )
    optimizer_ft = optim.SGD(
            model.parameters(),
            lr=opt.lr,
            weight_decay=5e-4,
            momentum=0.9,
            nesterov=True
        )

    part_fc = nn.Linear(256, class_split[opt.rank], bias=False)
    optimizer_part_fc = optim.SGD(
            part_fc.parameters(),
            lr=opt.lr,
            weight_decay=5e-4,
            momentum=0.9,
            nesterov=True
        )


    if opt.fp16 and opt.distributed:
        if opt.rank == 0:
            logging.info("distributed training with fp16 settings...")
        [model, part_fc], [optimizer_ft, optimizer_part_fc] = amp.initialize(
            [model.cuda(), part_fc.cuda()], [optimizer_ft, optimizer_part_fc], opt_level = "O1")
        # model, optimizer_ft = amp.initialize(model.cuda(), optimizer_ft, opt_level = "O1")

        # By default, apex.parallel.DistributedDataParallel overlaps communication with
        # computation in the backward pass.
        # model = DDP(model)
        # delay_allreduce delays all communication to the end of the backward pass.
        model = DDP(model, delay_allreduce=True)
        criterion = DistModelParallelCrossEntropy().cuda()

        train_model(opt, data_loader, sampler, model, part_fc, criterion, optimizer_ft, optimizer_part_fc, class_split)
        # train_model(opt, data_loader, sampler, model, criterion, optimizer_ft, class_split)
        # python -m torch.distributed.launch --nproc_per_node=2 main_amp.py -a resnet50 --b 224 --workers 4 --opt-level O2 ./
        # https://github.com/NVIDIA/apex/tree/master/examples/imagenet
