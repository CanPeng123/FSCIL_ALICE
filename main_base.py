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

from alice.model.model_factory import Model
from alice.loss.criterion import AngularPenaltySMLoss
from alice.validation.validation import NCMValidation
from alice.dataloader.data_utils import *
from alice.train.train import *
from utils.utils import *


def get_command_line_parser():
    parser = argparse.ArgumentParser('arguments for training')
    parser.add_argument('--data_root', type=str, help='path to dataset directory')
    parser.add_argument('--exp_dir', type=str, help='path to experiment directory')
    parser.add_argument('--trial', type=str, default='1', help='trial id')

    parser.add_argument('--arch', default='resnet18', help='model name is used for training')
    parser.add_argument('--feat_dim', default=2048, type=int, help='feature dimension')
    parser.add_argument('--num_proj_layers', type=int, default=2, help='number of projection layer')
    parser.add_argument('--no_projector', action='store_true', help='Do not use projector in backbone network training.')

    parser.add_argument('--batch_size', type=int, default=512, help='batch_size')
    parser.add_argument('--batch_size_new', type=int, default=0, help='set 0 will use all the availiable training image for new')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--epochs_base', type=int, default=800, help='number of training epochs for base session')
    parser.add_argument('--epochs_new', type=int, default=100, help='number of training epochs for following increemntal sessions')

    parser.add_argument('--print_freq', default=10, type=int, help='print frequency')
    parser.add_argument('--eval_freq', default=5, type=int, help='evaluate model frequency')
    parser.add_argument('--save_freq', default=50, type=int, help='save model frequency')
    parser.add_argument('--resume', default=None, type=str, help='path to latest checkpoint')

    parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--milestones', nargs='+', type=int, default=[60, 70])
    parser.add_argument('--gamma', type=float, default=0.1)

    parser.add_argument('--loss_type', type=str, default='cosface', choices=['arcface', 'sphereface', 'cosface', 'cross_entropy'])
    parser.add_argument('--loss_s', type=float, default=30.0)
    parser.add_argument('--loss_m', type=float, default=0.4)

    parser.add_argument('--dataset', type=str, default='cub200', choices=['mini_imagenet', 'cub200', 'cifar100'])
    parser.add_argument('--seed', type=int, default=-1)

    parser.add_argument('--data_transform', action='store_true', help='Whether use 2 set of transformed data per input image to do calculation.')
    parser.add_argument('--data_rotation', action='store_true', help='Rotate the input data to increase the categories.')
    parser.add_argument('--data_fusion', action='store_true', help='Fuse the input data to increase the categories.')

    parser.add_argument('--knowledge_distillation', action='store_true', help='Update the incremental model through knowledge distillation.')

    return parser


def main():
    # ---------- setup args ----------
    parser = get_command_line_parser()
    args = parser.parse_args()
    if not path.exists(args.exp_dir):
        makedirs(args.exp_dir)
    trial_dir = path.join(args.exp_dir, args.trial)
    if not path.exists(trial_dir):
        makedirs(trial_dir)
    tensorboard_dir = path.join(trial_dir, 'tensorboard')
    if not path.exists(tensorboard_dir):
        makedirs(tensorboard_dir)
    logger = SummaryWriter(tensorboard_dir)
    if args.seed != -1:
        print('set seed!')
        set_seed(args.seed)
    args = set_up_datasets(args)
    print(vars(args))

    # ---------- create model, optimizer & loss criterion ----------
    if args.dataset == 'cub200':
        model = Model(args, pretrained=True)
    else:
        model = Model(args, pretrained=False)

    angular_criterion = AngularPenaltySMLoss(loss_type=args.loss_type, s=args.loss_s, m=args.loss_m)

    model = model.cuda()
    angular_criterion = angular_criterion.cuda()
    cudnn.benchmark = True

    # ---------- load checkpoint ----------
    if args.resume is not None:
        if path.isfile(args.resume):
            start_epoch, model, optimizer = load_checkpoint(model, optimizer, args.resume)
            print("Loaded checkpoint '{}' (epoch {})".format(args.resume, start_epoch))
        else:
            print("No checkpoint found at '{}'".format(args.resume))

    # ---------- setup recording dict ----------
    best_acc = [0.0] * args.sessions
    best_acc_epoch = [0] * args.sessions

    # ---------- create validation  ----------
    print('creating NCM validation ...')
    validation_ncm = NCMValidation(args, model.encoder)

    # ---------- routine ----------
    for session in range(args.sessions):

        train_set, train_loader = get_train_dataloader(args, session)

        if session == 0:

            print('training base session {0} ...'.format(session))
            optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, nesterov=True, weight_decay=args.weight_decay)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)

            for epoch in range(args.epochs_base):
                lrc = scheduler.get_last_lr()[0]

                if args.data_transform:
                    train_loss = train_two_image(session, train_loader, model, angular_criterion, optimizer, epoch, args)
                else:
                    train_loss = train_one_image(session, train_loader, model, angular_criterion, optimizer, epoch, args)
                logger.add_scalar('Session {0} - Loss/train'.format(session), train_loss, epoch)
                print("Training... | epoch: {0} | learning rate: {1} | loss: {2}".format(epoch, lrc, train_loss))

                if epoch % args.eval_freq == 0:
                    print("Validating...")
                    val_ncm_acc = validation_ncm.eval()
                    print('Session 0 | NCM accuracy: {0} '.format(val_ncm_acc))
                    logger.add_scalar('Session {0} - Acc/val_ncm'.format(session), val_ncm_acc, epoch)
                    if val_ncm_acc > best_acc[session]:  # save the best model
                        best_acc[session] = val_ncm_acc
                        best_acc_epoch[session] = epoch
                        save_checkpoint(args, epoch, model, optimizer, None, None, val_ncm_acc, path.join(
                            trial_dir, 'trial{0}_session{1}_best.pth'.format(args.trial, session)), 'Saving the best model!')
                        print('Best ncm accuracy!!!')
                        best_model_dict = deepcopy(model.state_dict())

                if epoch % args.save_freq == 0:  # save the model
                    save_checkpoint(args, epoch, model, optimizer, None, None, val_ncm_acc,
                                    path.join(trial_dir, 'trial{0}_session{1}_ckpt_epoch_{2}.pth'.
                                              format(args.trial, session, epoch)), 'Saving...')

                logger.add_scalar('Session {0} - Learning rate/train'.format(session), lrc, epoch)
                scheduler.step()

            save_checkpoint(args, epoch, model, optimizer, None, None, val_ncm_acc,
                            path.join(trial_dir, 'trial{0}_session{1}_last.pth'.format(args.trial, session)),
                            'Saving the model at the last epoch.')

        else:
            print('training incremental session {0} ...'.format(session))

    print('Best top1 accuracy:', best_acc)
    print('Best top1 accuracy epoch:', best_acc_epoch)

    result_save_path = path.join(trial_dir, 'result.txt')
    txt_file = open(result_save_path, mode='w')
    txt_file.write('Best top1 accuracy:={}\n'.format(best_acc))
    txt_file.write('Best top1 accuracy epoch:{}\n'.format(best_acc_epoch))


if __name__ == '__main__':
    main()



