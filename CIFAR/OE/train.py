# -*- coding: utf-8 -*-
"""
Trains a WideResNet against near-distribution examples.
"""

import numpy as np
import sys
import os
import pickle
import argparse
import math
import time
from bisect import bisect_left
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as trn
import torchvision.datasets as dset
import torch.nn.functional as F
from torch.autograd import Variable as V

# go through rigamaroo to do ...utils.display_results import show_performance
if __package__ is None:
    import sys
    from os import path

    sys.path.append(path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))
    from utils.display_results import show_performance
    from utils.cifar_resnet import WideResNet
    from utils.log_sum_exp import log_sum_exp
    from utils.tinyimages_80mn_loader import TinyImages

np.random.seed(1)

parser = argparse.ArgumentParser(description='Trains a WideResNet against near-distribution examples',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# Positional arguments
parser.add_argument('dataset', type=str, choices=['cifar10', 'cifar100'],
                    help='Choose between CIFAR-10, CIFAR-100.')
parser.add_argument('--dontusetiny', action='store_true', help='Train against CIFAR images instead of 80mn.')
# Optimization options
parser.add_argument('--epochs', '-e', type=int, default=10, help='Number of epochs to train.')
parser.add_argument('--batch_size', '-b', type=int, default=256, help='Batch size.')
parser.add_argument('--learning_rate', '-lr', type=float, default=0.003, help='The initial learning rate.')
parser.add_argument('--momentum', '-m', type=float, default=0.9, help='Momentum.')
parser.add_argument('--decay', '-d', type=float, default=0.0005, help='Weight decay (L2 penalty).')
parser.add_argument('--test_bs', type=int, default=256)
# Checkpoints
parser.add_argument('--save', '-s', type=str, default='./snapshots', help='Folder to save checkpoints.')
parser.add_argument('--load', '-l', type=str, default='./snapshots', help='Checkpoint path to resume / test.')
parser.add_argument('--test', '-t', action='store_true', help='Test only flag.')
# Architecture
parser.add_argument('--layers', default=40, type=int, help='total number of layers (default: 28)')
parser.add_argument('--widen-factor', default=4, type=int, help='widen factor (default: 10)')
parser.add_argument('--droprate', default=0.3, type=float, help='dropout probability (default: 0.0)')
# Acceleration
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
parser.add_argument('--prefetch', type=int, default=2, help='Pre-fetching threads.')
args = parser.parse_args()


state = {k: v for k, v in args._get_kwargs()}
state['tt'] = 0     # SGDR variable
state['init_learning_rate'] = args.learning_rate


# mean and standard deviation of channels of CIFAR-10 images
mean = [x / 255 for x in [125.3, 123.0, 113.9]]
std = [x / 255 for x in [63.0, 62.1, 66.7]]

train_transform = trn.Compose(
    [trn.RandomCrop(32, padding=4), trn.RandomHorizontalFlip(), trn.ToTensor(),
     trn.Normalize(mean, std)])
test_transform = trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)])


train10 = dset.CIFAR10('/share/data/vision-greg/cifarpy', train=True, transform=train_transform, download=False)
train100 = dset.CIFAR100('/share/data/vision-greg/cifarpy', train=True, transform=train_transform, download=False)

test10 = dset.CIFAR10('/share/data/vision-greg/cifarpy', train=False, transform=test_transform, download=False)
test100 = dset.CIFAR100('/share/data/vision-greg/cifarpy', train=False, transform=test_transform, download=False)

if args.dataset == 'cifar10': num_classes = 10
else: num_classes = 100

tiny_images = TinyImages(transform=trn.Compose(
    [trn.ToTensor(), trn.ToPILImage(), trn.RandomCrop(32, padding=4),
     trn.RandomHorizontalFlip(), trn.ToTensor(), trn.Normalize(mean, std)]))


class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)


if args.dataset == 'cifar10':
    if args.dontusetiny is True:
        train_loader = torch.utils.data.DataLoader(
            ConcatDataset(
                train10, train100
            ),
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.prefetch, pin_memory=True)
    else:
        train_loader = torch.utils.data.DataLoader(
            ConcatDataset(
                train10, tiny_images
            ),
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.prefetch, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        test10,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.prefetch, pin_memory=True)
else:
    if args.dontusetiny is True:
        train_loader = torchdata.DataLoader(
            ConcatDataset(
                train100, train10
            ),
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.prefetch, pin_memory=True)
    else:
        train_loader = torch.utils.data.DataLoader(
            ConcatDataset(
                train100, tiny_images
            ),
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.prefetch, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        test100,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.prefetch, pin_memory=True)


# Create model
net = WideResNet(args.layers, num_classes, args.widen_factor, dropRate=args.droprate)

if args.load != '':
    model_name = os.path.join(args.load, args.dataset + '_model_epoch99.pytorch')
    if os.path.isfile(model_name):
        net.load_state_dict(torch.load(model_name))
        print('Model restored!')
    else:
        assert False, "model not restored"

if args.ngpu > 1:
    net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))

if args.ngpu > 0:
    net.cuda()
    torch.cuda.manual_seed(1)

cudnn.benchmark = True  # fire on all cylinders

optimizer = torch.optim.SGD(net.parameters(), state['learning_rate'], momentum=state['momentum'],
                            weight_decay=state['decay'], nesterov=True)


def train():
    net.train()  # enter train mode
    loss_avg = 0.0

    for batch_idx, (in_set, out_set) in enumerate(train_loader):
        data = torch.cat((in_set[0], out_set[0]), 0)
        target = in_set[1]

        data, target = V(data.cuda()), V(target.cuda())
        # forward
        x = net(data)
        # backward
        optimizer.zero_grad()
        loss = F.cross_entropy(x[:len(in_set[0])], target)
        # cross-entropy from softmax distribution to uniform distribution
        loss += 0.5 * -(x[len(in_set[0]):].mean(1) - log_sum_exp(x[len(in_set[0]):], dim=1)).mean()

        # # online hard example mining
        # scores = 0.5 * -torch.mean(torch.log(smax), dim=1)
        # _, hard_indices = scores.topk(64)
        # loss += scores[hard_indices].mean()

        loss.backward()
        optimizer.step()

        # exponential moving average
        loss_avg = loss_avg * 0.8 + loss.data[0] * 0.2

        dt = math.pi / float(args.epochs)
        state['tt'] += float(dt) / (len(train_loader.dataset) / float(args.batch_size))
        if state['tt'] >= math.pi - 0.01:
            state['tt'] = math.pi - 0.01
        curT = math.pi / 2.0 + state['tt']
        new_lr = args.learning_rate * (1.0 + math.sin(curT)) / 2.0  # lr_min = 0, lr_max = lr
        state['learning_rate'] = new_lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['learning_rate']

    state['train_loss'] = loss_avg


# test function
def test():
    net.eval()
    loss_avg = 0.0
    correct = 0
    for batch_idx, (data, target) in enumerate(test_loader):

        data, target = V(data.cuda(), volatile=True), V(target.cuda(), volatile=True)

        # forward
        x = net(data)

        loss = F.cross_entropy(x, target)
        pred = x.data.max(1)[1]
        correct += pred.eq(target.data).sum()

        # test loss average
        loss_avg += loss.data[0]

    state['test_loss'] = loss_avg / len(test_loader)
    state['test_accuracy'] = correct / len(test_loader.dataset)


if args.test:
    test()
    print(state)
    exit()

state['learning_rate'] = state['init_learning_rate']

print('Beginning Training')
# Main loop
best_accuracy = 0.0
for epoch in range(0, args.epochs):
    for param_group in optimizer.param_groups:
        param_group['lr'] = state['learning_rate']
    state['tt'] = math.pi / float(args.epochs) * epoch

    state['epoch'] = epoch

    begin_epoch = time.time()
    train()
    print('Epoch', epoch, '| Time Spent:', round(time.time() - begin_epoch, 4))

    test()

    torch.save(net.state_dict(), os.path.join(args.save, args.dataset + '_model_tuned_epoch' + str(epoch) + '.pytorch'))
    # Let us not waste space and delete the previous model
    # We do not overwrite the model because we need the epoch number
    try: os.remove(os.path.join(args.save, args.dataset + '_model_tuned_epoch' + str(epoch - 1) + '.pytorch'))
    except: True

    print(state)
