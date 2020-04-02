import os
import time
import logging
import argparse
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms

from apex.fp16_utils import *
from apex import amp, optimizers

from model import ft_net
from cross_entropy import ModelParallelCrossEntropy
from utils import get_class_split, get_onehot_label, compute_batch_acc


def get_data_loader(data_path, batch_size):
    transform_train_list = [
        transforms.Resize((384, 192), interpolation=3),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
    data_transforms = transforms.Compose(transform_train_list)
    image_dataset = datasets.ImageFolder(data_path, data_transforms)
    dataloader = torch.utils.data.DataLoader(
                                    image_dataset,
                                    batch_size=batch_size,
                                    shuffle=True,
                                    num_workers=8,
                                    pin_memory=True
                                    )
    return len(image_dataset.classes), dataloader


def train_model(opt, data_loader, model, criterion, optimizer, class_split):
    logging.info("Start training...")
    for epoch in range(opt.num_epochs):
        data_loader_iter = iter(data_loader)
        for step in range(len(data_loader)):
            start_time = time.time()
            images, labels = data_loader_iter.next()
            images = images.cuda(0)
            labels = labels.cuda(0)
            onehot_labels = get_onehot_label(labels, opt.num_gpus, opt.num_classes, opt.model_parallel, class_split)
            # Forward
            optimizer.zero_grad()
            logits = model(images, labels=onehot_labels)
            # Loss calculation
            if opt.model_parallel:
                # compute_loss = step > 0 and step % 10 == 0
                compute_loss = True
                loss = criterion(compute_loss, onehot_labels, *logits)
            else:
                loss = criterion(logits, labels)
            # Backward
            # with amp.scale_loss(loss, optimizer) as scaled_loss:
            #    scaled_loss.backward()
            loss.backward()
            optimizer.step()
            # Log training progress
            if step > 0 and step % 10 == 0:
                example_per_second = opt.batch_size / float(time.time() - start_time)
                batch_acc = compute_batch_acc(logits, labels, opt.batch_size, opt.model_parallel, step)
                logging.info(
                        "epoch [%.3d] iter = %d loss = %.3f acc = %.4f example/sec = %.2f" %
                        (epoch+1, step, loss.item(), batch_acc, example_per_second)
                    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s %(filename)s] %(message)s")

    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--gpus', default='0,1,2,3', type=str, help='0,1,2,3')
    parser.add_argument('--data_path', default='/your/data/path/Market-1501/train', type=str, help='training data path')
    parser.add_argument('--num_epochs', default=15, type=int, help='number of epochs')
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    parser.add_argument('--num_classes', default=0, type=int, help='number of classes')
    parser.add_argument('--lr', default=0.05, type=float, help='learning rate')
    parser.add_argument('--am', action="store_true", help='use am-softmax')
    parser.add_argument('--model_parallel', action="store_true", help='use model parallel')
    opt = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
    cudnn.benchmark = True
    gpu_ids = opt.gpus.split(",")
    num_gpus = len(gpu_ids)
    opt.num_gpus = num_gpus

    num_classes, data_loader = get_data_loader(opt.data_path, opt.batch_size)
    if opt.num_classes < num_classes:
        opt.num_classes = num_classes

    class_split = None
    if opt.model_parallel:
        # If using model parallel, split the number of classes
        # accroding to the number of GPUs
        class_split = get_class_split(opt.num_classes, num_gpus)

    model = ft_net(
            feature_dim=256,
            num_classes=opt.num_classes,
            num_gpus=num_gpus,
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

    if opt.model_parallel:
        # When using model parallel, we wrap all the model except classifier in DataParallel
        model, optimizer_ft = amp.initialize(model, optimizer_ft, opt_level = "O1")
        model.backbone = nn.DataParallel(model.backbone).cuda()
        model.features = nn.DataParallel(model.features).cuda()
        criterion = ModelParallelCrossEntropy().cuda()
    else:
        # When not using model parallel, we use DataParallel directly
        model, optimizer_ft = amp.initialize(model, optimizer_ft, opt_level = "O1")
        model = nn.DataParallel(model).cuda()
        criterion = nn.CrossEntropyLoss().cuda()

    train_model(opt, data_loader, model, criterion, optimizer_ft, class_split)
