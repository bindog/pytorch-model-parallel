import torch
import torch.cuda.comm as comm


def get_class_split(num_classes, num_gpus):
    class_split = []
    for i in range(num_gpus):
        _class_num = num_classes // num_gpus
        if i < (num_classes % num_gpus):
            _class_num += 1
        class_split.append(_class_num)
    return class_split


def get_onehot_label(labels, num_gpus, num_classes, model_parallel=False, class_split=None):
    # Get one-hot labels
    labels = labels.view(-1, 1)
    labels_onehot = torch.zeros(len(labels), num_classes).cuda()
    labels_onehot.scatter_(1, labels, 1)

    if not model_parallel:
        return labels_onehot
    else:
        label_tuple = comm.scatter(labels_onehot, range(num_gpus), class_split, dim=1)
        return label_tuple


def compute_batch_acc(outputs, labels, batch_size, model_parallel, step):
    if model_parallel:
        if not (step > 0 and step % 10 == 0):
            return 0
        outputs = [outputs]
        max_score = None
        max_preds = None
        base = 0
        for logit_same_tuple in zip(*outputs):
            _split = logit_same_tuple[0].size()[1]
            score, preds = torch.max(sum(logit_same_tuple).data, dim=1)
            score = score.to(0)
            preds = preds.to(0)
            if max_score is not None:
                cond = score > max_score
                max_preds = torch.where(cond, preds + base, max_preds)
                max_score = torch.where(cond, score, max_score)
            else:
                max_score = score
                max_preds = preds
            base += _split
        preds = max_preds
        batch_acc = torch.sum(preds == labels).item() / batch_size
    else:
        _, preds = torch.max(outputs.data, 1)
        batch_acc = torch.sum(preds == labels).item() / batch_size

    return batch_acc


if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4"
    labels = torch.tensor([5, 2, 3, 4, 6, 9, 7, 1]).cuda()
    label_tuple = get_onehot_label(labels, 4, 12, [3, 3, 3, 3])
    for label in label_tuple:
        print(label.size())
        print(label)
