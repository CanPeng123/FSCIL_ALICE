#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import builtins
import os
import random
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
from tqdm import tqdm
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import confusion_matrix
import matplotlib.ticker as ticker
from alice.dataloader.data_utils import *


import warnings
warnings.filterwarnings("ignore")

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

def get_command_line_parser():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--data_root', type=str, help='path to dataset directory')
    parser.add_argument('--exp_dir', type=str, help='path to experiment directory')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50', choices=model_names,
                        help='model architecture: ' + ' | '.join(model_names) + ' (default: resnet50)')
    parser.add_argument('--num_cls', default=10, type=int, metavar='N',
                        help='number of classes in dataset (output dimention of models)')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 32)')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch_size', default=256, type=int, metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                             'batch size of all GPUs on the current node when '
                             'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--lr', '--learning_rate', default=30., type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--schedule', default=[60, 80], nargs='*', type=int,
                        help='learning rate schedule (when to drop lr by a ratio)')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight_decay', default=0., type=float,
                        metavar='W', help='weight decay (default: 0.)', dest='weight_decay')
    parser.add_argument('-p', '--print-freq', default=500, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int, help='node rank for distributed training')
    parser.add_argument('--dist_url', default='tcp://224.66.41.62:23456', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or'
                             'multi node data parallel training')

    parser.add_argument('--pretrained', default='', type=str, help='path to pretrained checkpoint')
    parser.add_argument('--trial', type=str, default='1', help='trial id')

    parser.add_argument('--dataset', type=str, default='cub200', choices=['mini_imagenet', 'cub200', 'cifar100'])
    parser.add_argument('--current_session', default=0, type=int)
    parser.add_argument('--used_img', default=500, type=int)  # 500, 5, 1
    parser.add_argument('--balanced', default=0, type=int)

    return parser


def get_backbone(dataset, backbone_name, num_cls=10):
    if dataset == 'cifar100' or dataset == 'mini_imagenet':
        from alice.model.resnet_CIFAR import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
    elif dataset == 'cub200':
        from alice.model.resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
    else:
        raise RuntimeError('Something is wrong.')

    models = {'resnet18': ResNet18(low_dim=num_cls),
              'resnet34': ResNet34(low_dim=num_cls),
              'resnet50': ResNet50(low_dim=num_cls),
              'resnet101': ResNet101(low_dim=num_cls),
              'resnet152': ResNet152(low_dim=num_cls)}
    return models[backbone_name]


best_acc1 = 0
def main():
    parser = get_command_line_parser()
    args = parser.parse_args()
    args = set_up_datasets(args)
    print(vars(args))

    if not os.path.exists(args.exp_dir):
        os.makedirs(args.exp_dir)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    print("=> creating model '{}'".format(args.arch))
    # model = models.__dict__[args.arch]()
    model = get_backbone(args.dataset, args.arch, args.num_cls)

    # freeze all layers but the last fc
    for name, param in model.named_parameters():
        if name not in ['fc.weight', 'fc.bias']:
            param.requires_grad = False
    # init the fc layer
    model.fc.weight.data.normal_(mean=0.0, std=0.01)
    model.fc.bias.data.zero_()

    # load from pre-trained, before DistributedDataParallel constructor
    if args.pretrained:
        if os.path.isfile(args.pretrained):
            print("=> loading checkpoint '{}'".format(args.pretrained))
            checkpoint = torch.load(args.pretrained, map_location="cpu")
            state_dict = checkpoint['state_dict']

            new_state_dict = dict()
            for old_key, value in state_dict.items():
                if old_key.startswith('backbone') and 'fc' not in old_key:
                    new_key = old_key.replace('backbone.', '')
                    new_state_dict[new_key] = value

            args.start_epoch = 0
            msg = model.load_state_dict(new_state_dict, strict=False)
            assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}

            print("=> loaded pre-trained model '{}'".format(args.pretrained))
        else:
            print("=> no checkpoint found at '{}'".format(args.pretrained))

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    cudnn.benchmark = True

    # -------------------------------------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------------------------------------

    # load training data
    trainset, train_loader, testset, testloader = get_incremental_dataset_fs(args, session=args.current_session)
    print('length of the trainset: {0}'.format(len(trainset)))

    # ----------------------------- calculate and store average class-wise feature embedding -----------------------------
    print('----------------------------- calculate and store average class-wise feature embedding -----------------------------')
    cls_wise_feature_prototype = []
    cls_label = []

    transform = testloader.dataset.transform
    cls_avg_feature, cls_avg_feature_index = calculate_avg_feature_for_each_cls(trainset, transform, model, args)

    for i in range(len(cls_avg_feature)):
        cls_wise_feature_prototype.append(cls_avg_feature[i])
        cls_label.append(cls_avg_feature_index[i])
    feature_save_dir = os.path.join(args.exp_dir, 'cls_wise_avg_feature.pth')
    torch.save(dict(class_feature=cls_wise_feature_prototype, class_id=cls_label), feature_save_dir)

    # ----------------------------- do interference -----------------------------
    print('----------------------------- do interference -----------------------------')
    save_path = os.path.join(args.exp_dir, 'result.txt')
    prediction_result, label_list = test_NCM(model, testloader, args, cls_wise_feature_prototype, save_path)


def calculate_avg_feature_for_each_cls(trainset, transform, model, args):
    model = model.eval()

    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=128, num_workers=8, pin_memory=True, shuffle=False)
    trainloader.dataset.transform = transform

    overall_avg_feature = []
    overall_avg_cls = []
    final_avg_feature = []
    final_avg_cls = []
    embedding_list = []
    label_list = []

    with torch.no_grad():
        tqdm_gen = tqdm(trainloader)
        for _, batch in enumerate(tqdm_gen, 1):
            data, label = [_.cuda() for _ in batch]
            embedding = model.encode(data)
            embedding_list.append(embedding.cpu())
            label_list.append(label.cpu())
    embedding_list = torch.cat(embedding_list, dim=0)
    label_list = torch.cat(label_list, dim=0)

    # generate the average feature with all data
    for index in range(args.num_cls):
        class_index = (label_list == index).nonzero()
        embedding_this = embedding_list[class_index.squeeze(-1)]
        embedding_this = embedding_this.mean(0, keepdims=True).cuda()
        overall_avg_feature.append(embedding_this)
        overall_avg_cls.append(index)


    if (args.balanced == 1 and args.used_img == 500) or (args.balanced == 0 and args.used_img == 500):  # using all data
        for index in range(args.num_cls):
            final_avg_feature.append(overall_avg_feature[index])
            final_avg_cls.append(overall_avg_cls[index])
    elif args.balanced == 1 and args.used_img == 5:  # using balanced 5 data
        for index in range(args.num_cls):
            top_5_close_feature = []
            top_5_distance = []
            class_index = (label_list == index).nonzero()
            embedding_this = embedding_list[class_index.squeeze(-1)]
            if index < args.base_class:
                for i in range(len(embedding_this)):
                    if i < 5:
                        embedding_this_buff = embedding_this[i].view(-1, embedding_this[i].size()[0])
                        top_5_close_feature.append(embedding_this_buff)
                        distance = pairwise_distances(np.asarray(embedding_this_buff.cpu()), np.asarray(overall_avg_feature[index].cpu()), metric='cosine')
                        top_5_distance.append(distance)
                    else:
                        embedding_this_buff = embedding_this[i].view(-1, embedding_this[i].size()[0])
                        distance = pairwise_distances(np.asarray(embedding_this_buff.cpu()), np.asarray(overall_avg_feature[index].cpu()), metric='cosine')
                        for j in range(len(top_5_distance)):
                            if distance < top_5_distance[j]:
                                top_5_close_feature[j] = embedding_this_buff
                                top_5_distance[j] = distance
                                break
            else:
                for i in range(5):
                    embedding_this_buff = embedding_this[i].view(-1, embedding_this[i].size()[0])
                    top_5_close_feature.append(embedding_this_buff)
            top_5_close_feature = torch.cat(top_5_close_feature, dim=0)
            top_5_close_feature = top_5_close_feature.mean(0, keepdims=True).cuda()
            final_avg_feature.append(top_5_close_feature)
            final_avg_cls.append(overall_avg_cls[index])
    elif args.balanced == 1 and args.used_img == 1:  # using balanced 1 data
        for index in range(args.num_cls):
            top_1_close_feature = []
            top_1_distance = []
            class_index = (label_list == index).nonzero()
            embedding_this = embedding_list[class_index.squeeze(-1)]
            if index < args.base_class:
                for i in range(len(embedding_this)):
                    if i < 1:
                        embedding_this_buff = embedding_this[i].view(-1, embedding_this[i].size()[0])
                        top_1_close_feature.append(embedding_this_buff)
                        distance = pairwise_distances(np.asarray(embedding_this_buff.cpu()), np.asarray(overall_avg_feature[index].cpu()), metric='cosine')
                        top_1_distance.append(distance)
                    else:
                        embedding_this_buff = embedding_this[i].view(-1, embedding_this[i].size()[0])
                        distance = pairwise_distances(np.asarray(embedding_this_buff.cpu()), np.asarray(overall_avg_feature[index].cpu()), metric='cosine')
                        for j in range(len(top_1_distance)):
                            if distance < top_1_distance[j]:
                                top_1_close_feature[j] = embedding_this_buff
                                top_1_distance[j] = distance
                                break
            else:
                for i in range(1):
                    embedding_this_buff = embedding_this[i].view(-1, embedding_this[i].size()[0])
                    top_1_close_feature.append(embedding_this_buff)
            top_1_close_feature = torch.cat(top_1_close_feature, dim=0)
            top_1_close_feature = top_1_close_feature.mean(0, keepdims=True).cuda()
            final_avg_feature.append(top_1_close_feature)
            final_avg_cls.append(overall_avg_cls[index])

    return final_avg_feature, final_avg_cls



def acquire_backbone_feature(model, trainset, transform, start_cls=0, end_cls=99, maximum_feature_num_per_cls=500):
    model = model.eval()
    feature_list = []
    label_list = []
    count_list = []
    for i in range(start_cls, end_cls):
        count_list.append(0)

    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=1, num_workers=1, pin_memory=True, shuffle=False)
    trainloader.dataset.transform = transform
    print('acquiring backbone feature..................')

    with torch.no_grad():
        tqdm_gen = tqdm(trainloader)
        for _, batch in enumerate(tqdm_gen, 1):
            data, label = [_.cuda() for _ in batch]
            feature = model.encode(data)
            num_of_data = feature.size()[0]

            for i in range(num_of_data):
                for j in range(start_cls, end_cls):
                    if label[i] == j:
                        if count_list[j - start_cls] < maximum_feature_num_per_cls:
                            # feature[i] = torch.nn.functional.normalize(feature[i], p=2, dim=-1)
                            feature_list.append(feature[i])
                            label_list.append(label[i])
                            count_list[j - start_cls] = count_list[j - start_cls] + 1
    return feature_list, label_list


def test_NCM(model, testloader, args, cls_wise_feature_prototype, save_path):
    model = model.eval()

    embedding_list = []
    label_list = []

    with torch.no_grad():
        tqdm_gen = tqdm(testloader)
        for _, batch in enumerate(tqdm_gen, 1):
            data, label = [_.cuda() for _ in batch]
            embedding = model.encode(data)
            embedding_list.append(embedding.cpu())
            label_list.append(label.cpu())
    embedding_list = torch.cat(embedding_list, dim=0).cpu()
    embedding_list = torch.nn.functional.normalize(embedding_list, p=2, dim=-1)

    label_list = torch.cat(label_list, dim=0).cpu()

    for i in range(len(cls_wise_feature_prototype)):
        cls_wise_feature_prototype[i] = cls_wise_feature_prototype[i].view(-1)
    proto_list = torch.stack(cls_wise_feature_prototype, dim=0).cpu()
    proto_list = torch.nn.functional.normalize(proto_list, p=2, dim=-1)

    # metric: euclidean, cosine, l2, l1
    pairwise_distance = pairwise_distances(np.asarray(embedding_list), np.asarray(proto_list), metric='cosine')
    prediction_result = np.argmin(pairwise_distance, axis=1)

    label_list = np.asarray(label_list)
    total_acc = np.sum(prediction_result == label_list) / float(len(label_list))

    num_of_img_per_task = [0] * args.sessions
    correct_prediction_per_task = [0] * args.sessions
    acc_list = [0.0] * args.sessions

    for i in range(args.sessions):
        if i == 0:
            start_class = 0
            end_class = args.base_class
        else:
            start_class = args.base_class + (i - 1) * args.way
            end_class = args.base_class + i * args.way

        for k in range(len(label_list)):
            if start_class <= label_list[k] < end_class:
                num_of_img_per_task[i] = num_of_img_per_task[i] + 1
                if label_list[k] == prediction_result[k]:
                    correct_prediction_per_task[i] = correct_prediction_per_task[i] + 1

        if num_of_img_per_task[i] != 0:
            acc_list[i] = correct_prediction_per_task[i] / num_of_img_per_task[i]

    print('TEST, total average accuracy={:.4f}'.format(total_acc))
    print('TEST, task-wise correct prediction: {0}'.format(correct_prediction_per_task))
    print('TEST, task-wise number of images: {0}'.format(num_of_img_per_task))
    print('TEST, task-wise accuracy: {0}'.format(acc_list))

    if save_path != None:
        txt_file = open(save_path, mode='w')
        # txt_file.write('---------------- session: {0} --------------------------------------\n'.format(session))
        txt_file.write('TEST, total average accuracy={:.4f}\n'.format(total_acc))
        txt_file.write('TEST, task-wise correct prediction: {0}\n'.format(correct_prediction_per_task))
        txt_file.write('TEST, task-wise number of images: {0}\n'.format(num_of_img_per_task))
        txt_file.write('TEST, task-wise accuracy: {0}\n'.format(acc_list))

    return prediction_result, label_list



if __name__ == '__main__':
    main()
