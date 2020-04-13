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


def get_sparse_onehot_label(labels, num_gpus, num_classes, model_parallel=False, class_split=None):
    # Get sparse one-hot labels
    labels_list = labels.tolist()
    batch_size = len(labels_list)

    if not model_parallel:
        sparse_index = torch.LongTensor([*range(batch_size), labels_list])
        sparse_value = torch.ones(batch_size, dtype=torch.long)
        labels_onehot = torch.sparse.LongTensor(sparse_index, sparse_value, torch.Size([batch_size, num_classes]))
    else:
        assert num_gpus == len(class_split), "class split parts not equal to num of gpus!"
        # prepare dict for generating sparse tensor
        splits_dict = {}
        start_index = 0
        for i, num_splits in enumerate(class_split):
            splits_dict[i] = {}
            end_index = start_index + num_splits
            splits_dict[i]["start_index"] = start_index
            splits_dict[i]["num_splits"] = num_splits
            splits_dict[i]["end_index"] = end_index
            splits_dict[i]["index_list"] = []
            splits_dict[i]["nums"] = 0
            start_index = end_index
        # get valid index in each split
        for i, label in enumerate(labels_list):
            for j in range(num_gpus):
                if label >= splits_dict[j]["start_index"] and label < splits_dict[j]["end_index"]:
                    valid_index = [i, label - splits_dict[j]["start_index"]]
                    splits_dict[j]["index_list"].append(valid_index)
                    splits_dict[j]["nums"] += 1
                    break
        # finally get the sparse tensor
        label_tuple = []
        for i in range(num_gpus):
            if splits_dict[i]["nums"] == 0:
                sparse_tensor = torch.sparse.LongTensor(torch.Size([batch_size, splits_dict[i]["num_splits"]]))
                label_tuple.append(sparse_tensor.to(i))
            else:
                sparse_index = torch.LongTensor(splits_dict[i]["index_list"])
                sparse_value = torch.ones(splits_dict[i]["nums"], dtype=torch.long)
                sparse_tensor = torch.sparse.LongTensor(
                                                    sparse_index.t(),
                                                    sparse_value,
                                                    torch.Size([batch_size, splits_dict[i]["num_splits"]])
                                                )
                label_tuple.append(sparse_tensor.to(i))
        return tuple(label_tuple)


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
    # import os
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4"
    # labels = torch.tensor([5, 2, 3, 4, 6, 9, 7, 1]).cuda()
    # label_tuple = get_onehot_label(labels, 4, 12, [3, 3, 3, 3])
    # for label in label_tuple:
    #     print(label.size())
    #     print(label)
    labels = torch.tensor([5, 2, 3, 4, 6, 9, 7, 1])
    print(labels)
    label_tuple = get_sparse_onehot_label(labels, 4, 12, True, [3, 3, 3, 3])
    for label in label_tuple:
        print(label.size())
        print(label.to_dense())
