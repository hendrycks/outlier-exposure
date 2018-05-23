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
    import utils.svhn_loader as svhn
    from utils.cifar_resnet import WideResNet
    from utils.log_sum_exp import log_sum_exp
    from utils.tinyimages_80mn_loader import TinyImages

np.random.seed(1)

parser = argparse.ArgumentParser(description='Trains a WideResNet against near-distribution examples',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# Positional arguments
parser.add_argument('--dontusetiny', action='store_true', help='Train against CIFAR images instead of 80mn.')
# Optimization options
parser.add_argument('--epochs', '-e', type=int, default=5, help='Number of epochs to train.')
parser.add_argument('--batch_size', '-b', type=int, default=256, help='Batch size.')
parser.add_argument('--learning_rate', '-lr', type=float, default=0.001, help='The initial learning rate.')
parser.add_argument('--momentum', '-m', type=float, default=0.9, help='Momentum.')
parser.add_argument('--decay', '-d', type=float, default=0.0005, help='Weight decay (L2 penalty).')
parser.add_argument('--test_bs', type=int, default=256)
# Checkpoints
parser.add_argument('--save', '-s', type=str, default='./snapshots', help='Folder to save checkpoints.')
parser.add_argument('--load', '-l', type=str, default='./snapshots', help='Checkpoint path to resume / test.')
parser.add_argument('--test', '-t', action='store_true', help='Test only flag.')
# Architecture
parser.add_argument('--layers', default=16, type=int, help='total number of layers (default: 16)')
parser.add_argument('--widen-factor', default=4, type=int, help='widen factor (default: 4)')
parser.add_argument('--droprate', default=0.4, type=float, help='dropout probability (default: 0.0)')
# Acceleration
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
parser.add_argument('--prefetch', type=int, default=2, help='Pre-fetching threads.')
args = parser.parse_args()


state = {k: v for k, v in args._get_kwargs()}
state['tt'] = 0     # SGDR variable
state['init_learning_rate'] = args.learning_rate

train_data = svhn.SVHN(root='/share/data/vision-greg/svhn/', split="train_and_extra",
                       transform=trn.ToTensor(), download=False)
test_data = svhn.SVHN(root='/share/data/vision-greg/svhn/', split="test",
                      transform=trn.ToTensor(), download=False)

num_classes = 10

tiny_images = TinyImages(transform=trn.Compose(
    [trn.ToTensor(), trn.ToPILImage(),
     trn.RandomHorizontalFlip(), trn.ToTensor()]), exclude_cifar=True)


val_share = 5000/604388.


# split into train and validation set https://gist.github.com/t-vi/9f6118ff84867e89f3348707c7a1271f
class PartialDataset(torch.utils.data.Dataset):
    def __init__(self, parent_ds, offset, length):
        self.parent_ds = parent_ds
        self.offset = offset
        self.length = length
        assert len(parent_ds) >= offset + length, Exception("Parent Dataset not long enough")
        super(PartialDataset, self).__init__()

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        return self.parent_ds[i + self.offset]


def validation_split(dataset, val_share=0.1):
    """
       Split a (training and vaidation combined) dataset into training and validation.
       Note that to be statistically sound, the items in the dataset should be statistically
       independent (e.g. not sorted by class, not several instances of the same dataset that
       could end up in either set).

       inputs:
          dataset:   ("training") dataset to split into training and validation
          val_share: fraction of validation data (should be 0<val_share<1, default: 0.1)
       returns: input dataset split into test_ds, val_ds

    """
    val_offset = int(len(dataset) * (1 - val_share))
    return PartialDataset(dataset, 0, val_offset), PartialDataset(dataset, val_offset, len(dataset) - val_offset)


train_data, val_data = validation_split(train_data, val_share)
train_size = train_data.length


class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)


train_loader = torch.utils.data.DataLoader(
    ConcatDataset(
        train_data, tiny_images
    ),
    batch_size=args.batch_size, shuffle=True,
    num_workers=args.prefetch, pin_memory=True)

test_loader = torch.utils.data.DataLoader(
    test_data,
    batch_size=args.batch_size, shuffle=False,
    num_workers=args.prefetch, pin_memory=True)


# Create model
net = WideResNet(args.layers, num_classes, args.widen_factor, dropRate=args.droprate)

if args.ngpu > 1:
    net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))

if args.ngpu > 0:
    net.cuda()
    torch.cuda.manual_seed(1)

cudnn.benchmark = True  # fire on all cylinders

if args.load != '':
    model_name = os.path.join(args.load, 'svhn_model_epoch19.pytorch')
    if os.path.isfile(model_name):
        net.load_state_dict(torch.load(model_name))
        print('Model restored!')
    else:
        assert False, "model not restored"

optimizer = torch.optim.SGD(net.parameters(), state['learning_rate'], momentum=state['momentum'],
                            weight_decay=state['decay'], nesterov=True)

from tqdm import tqdm
def train():
    net.train()  # enter train mode
    loss_avg = 0.0
    for batch_idx, (in_set, out_set) in enumerate(tqdm(train_loader)):
        data = torch.cat((in_set[0], out_set[0]), 0)
        target = torch.squeeze(in_set[1].long())

        data, target = V(data.cuda()), V(target.cuda())

        # forward
        x = net(data)

        # backward
        optimizer.zero_grad()
        loss = F.cross_entropy(x[:len(in_set[0])], target)
        # cross-entropy from softmax distribution to uniform distribution
        loss += 0.5 * -(x[len(in_set[0]):].mean(1) - log_sum_exp(x[len(in_set[0]):], dim=1)).mean()

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

        data, target = V(data.cuda(), volatile=True), V(torch.squeeze(target).long().cuda(), volatile=True)

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

    mix_indicator = ''
    if args.mix:
        mix_indicator = '_mix'

    torch.save(net.state_dict(), os.path.join(args.save, 'svhn_model_tuned_epoch' + str(epoch) +
                                              mix_indicator + '.pytorch'))
    # Let us not waste space and delete the previous model
    # We do not overwrite the model because we need the epoch number
    try: os.remove(os.path.join(args.save, 'svhn_model_tuned_epoch' + str(epoch - 1) +
                                mix_indicator + '.pytorch'))
    except: True

    print(state)
