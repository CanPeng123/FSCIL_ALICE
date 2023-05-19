import argparse
import time
import math
import random
import os
from os import path, makedirs
from copy import deepcopy

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
from torch.backends import cudnn

from utils.utils import *


def fusion_aug_one_image(x, y, session, args, alpha=20.0, mix_times=4):  # mixup based
    batch_size = x.size()[0]
    mix_data = []
    mix_target = []

    # print('fusion_aug_one_image | before fusion | length of the data: {0}, size of image: {1}'.format(len(y), x.size()))
    for _ in range(mix_times):
        index = torch.randperm(batch_size).cuda()
        for i in range(batch_size):
            if y[i] != y[index][i]:
                new_label = fusion_aug_generate_label(y[i].item(), y[index][i].item(), session, args)
                lam = np.random.beta(alpha, alpha)
                if lam < 0.4 or lam > 0.6:
                    lam = 0.5
                mix_data.append(lam * x[i] + (1 - lam) * x[index, :][i])
                mix_target.append(new_label)

    new_target = torch.Tensor(mix_target)
    y = torch.cat((y, new_target.cuda().long()), 0)
    for item in mix_data:
        x = torch.cat((x, item.unsqueeze(0)), 0)
    # print('fusion_aug_one_image | after fusion | length of the data: {0}, size of image: {1}'.format(len(y), x.size()))

    return x, y


def fusion_aug_two_image(x_1, x_2, y, session, args, alpha=20.0, mix_times=4):  # mixup based
    batch_size = x_1.size()[0]
    mix_data_1 = []
    mix_data_2 = []
    mix_target = []

    # print('fusion_aug_two_image | before fusion | length of the data: {0}, size of image: {1}'.format(len(y), x_1.size()))
    for _ in range(mix_times):
        index = torch.randperm(batch_size).cuda()
        for i in range(batch_size):
            if y[i] != y[index][i]:
                new_label = fusion_aug_generate_label(y[i].item(), y[index][i].item(), session, args)
                lam = np.random.beta(alpha, alpha)
                if lam < 0.4 or lam > 0.6:
                    lam = 0.5
                mix_data_1.append(lam * x_1[i] + (1 - lam) * x_1[index, :][i])
                mix_data_2.append(lam * x_2[i] + (1 - lam) * x_2[index, :][i])
                mix_target.append(new_label)

    new_target = torch.Tensor(mix_target)
    y = torch.cat((y, new_target.cuda().long()), 0)
    for item in mix_data_1:
        x_1 = torch.cat((x_1, item.unsqueeze(0)), 0)
    for item in mix_data_2:
        x_2 = torch.cat((x_2, item.unsqueeze(0)), 0)
    # print('fusion_aug_two_image | after fusion | length of the data: {0}, size of image: {1}'.format(len(y), x_1.size()))

    return x_1, x_2, y


def fusion_aug_generate_label(y_a, y_b, session, args):
    current_total_cls_num = args.base_class + session * args.way
    if session == 0:  # base session -> increasing: [(args.base_class) * (args.base_class - 1)]/2
        y_a, y_b = y_a, y_b
        assert y_a != y_b
        if y_a > y_b:  # make label y_a smaller than y_b
            tmp = y_a
            y_a = y_b
            y_b = tmp
        label_index = ((2 * current_total_cls_num - y_a - 1) * y_a) / 2 + (y_b - y_a) - 1
    else:  # incremental session -> increasing: [(args.way) * (args.way - 1)]/2
        y_a = y_a - (current_total_cls_num - args.way)
        y_b = y_b - (current_total_cls_num - args.way)
        assert y_a != y_b
        if y_a > y_b:  # make label y_a smaller than y_b
            tmp = y_a
            y_a = y_b
            y_b = tmp
        label_index = int(((2 * args.way - y_a - 1) * y_a) / 2 + (y_b - y_a) - 1)
    return label_index + current_total_cls_num


def train_one_image(session, train_loader, model, angular_criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(len(train_loader), [batch_time, losses], prefix="Session: {0} | Epoch: [{1}]".format(session, epoch))

    model.train()  # switch to train mode

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        images = images.cuda()
        target = target.cuda()
        # print('train_one_image | length of the data: {0}, size of image: {1}'.format(len(target), images.size()))

        if args.data_fusion:
            images, target = fusion_aug_one_image(images, target, session, args, alpha=20.0, mix_times=4)

        loss = 0
        # compute angular penalty
        features = model.get_angular_output(images)
        angular_loss = angular_criterion(features, target)
        # print('angular_loss: {0}'.format(angular_loss))
        loss = loss + angular_loss

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        losses.update(loss.item(), images.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

    return losses.avg


def train_two_image(session, train_loader, model, angular_criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(len(train_loader), [batch_time, losses], prefix="Session: {0} | Epoch: [{1}]".format(session, epoch))
    model.train()  # switch to train mode

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        images[0] = images[0].cuda()
        images[1] = images[1].cuda()
        target = target.cuda()
        # print('train_two_image | length of the data: {0}, size of image: {1}'.format(len(target), images[0].size()))

        if args.data_fusion:
            images[0], images[1], target = fusion_aug_two_image(images[0], images[1], target, session, args)

        loss = 0
        # compute angular penalty
        arcface_output_1 = model.get_angular_output(images[0])
        arcface_output_2 = model.get_angular_output(images[1])
        angular_loss_1 = angular_criterion(arcface_output_1, target)
        angular_loss_2 = angular_criterion(arcface_output_2, target)
        angular_loss = 0.5 * angular_loss_1 + 0.5 * angular_loss_2
        # print('angular_loss: {0}'.format(angular_loss))
        loss = loss + angular_loss

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        losses.update(loss.item(), images[0].size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

    return losses.avg
