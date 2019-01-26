# -*- coding: utf-8 -*-
"""
Trains a MNIST classifier.
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
import torchtext

from torchtext import data
from torchtext import datasets

import tqdm



np.random.seed(1)

parser = argparse.ArgumentParser(description='Train without OE',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--in_dist_dataset', type=str, choices=['sst', '20ng', 'trec'], default='sst')
# Optimization options
parser.add_argument('--epochs', '-e', type=int, default=5, help='Number of epochs to train.')
parser.add_argument('--batch_size', '-b', type=int, default=64, help='Batch size.')
parser.add_argument('--learning_rate', '-lr', type=float, default=0.01, help='The initial learning rate.')
parser.add_argument('--momentum', '-m', type=float, default=0.5, help='Momentum.')
parser.add_argument('--test_bs', type=int, default=256)
# Checkpoints
parser.add_argument('--save', '-s', type=str, default='./snapshots', help='Folder to save checkpoints.')
parser.add_argument('--load', '-l', type=str, default='./snapshots', help='Checkpoint path to resume / test.')
parser.add_argument('--test', '-t', action='store_true', help='Test only flag.')
parser.add_argument('--mix', dest='mix', action='store_true', help='Mix outliers images with in-dist images.')
# Acceleration
parser.add_argument('--prefetch', type=int, default=2, help='Pre-fetching threads.')
args = parser.parse_args()


if args.in_dist_dataset == 'sst':
    # set up fields
    TEXT = data.Field(pad_first=True)
    LABEL = data.Field(sequential=False)

    # make splits for data
    train, val, test = datasets.SST.splits(
        TEXT, LABEL, fine_grained=False, train_subtrees=False,
        filter_pred=lambda ex: ex.label != 'neutral')

    # build vocab
    TEXT.build_vocab(train, max_size=10000)
    LABEL.build_vocab(train, max_size=10000)
    print('vocab length (including special tokens):', len(TEXT.vocab))

    # create our own iterator, avoiding the calls to build_vocab in SST.iters
    train_iter, val_iter, test_iter = data.BucketIterator.splits(
        (train, val, test), batch_size=args.batch_size, repeat=False)
elif args.in_dist_dataset == '20ng':
    TEXT = data.Field(pad_first=True, lower=True, fix_length=100)
    LABEL = data.Field(sequential=False)

    train = data.TabularDataset(path='./.data/20newsgroups/20ng-train.txt',
                                     format='csv',
                                     fields=[('label', LABEL), ('text', TEXT)])

    test = data.TabularDataset(path='./.data/20newsgroups/20ng-test.txt',
                                     format='csv',
                                     fields=[('label', LABEL), ('text', TEXT)])

    TEXT.build_vocab(train, max_size=10000)
    LABEL.build_vocab(train, max_size=10000)
    print('vocab length (including special tokens):', len(TEXT.vocab))

    train_iter = data.BucketIterator(train, batch_size=args.batch_size, repeat=False)
    test_iter = data.BucketIterator(test, batch_size=args.batch_size, repeat=False)
elif args.in_dist_dataset == 'trec':
    # set up fields
    TEXT = data.Field(pad_first=True, lower=True)
    LABEL = data.Field(sequential=False)

    # make splits for data
    train, test = datasets.TREC.splits(TEXT, LABEL, fine_grained=True)


    # build vocab
    TEXT.build_vocab(train, max_size=10000)
    LABEL.build_vocab(train, max_size=10000)
    print('vocab length (including special tokens):', len(TEXT.vocab))
    print('num labels:', len(LABEL.vocab))

    # make iterators
    train_iter, test_iter = data.BucketIterator.splits(
        (train, test), batch_size=args.batch_size, repeat=False)





cudnn.benchmark = True  # fire on all cylinders


class ClfGRU(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(len(TEXT.vocab), 50, padding_idx=1)
        self.gru = nn.GRU(input_size=50, hidden_size=128, num_layers=2,
            bias=True, batch_first=True, bidirectional=False)
        self.linear = nn.Linear(128, num_classes)

    def forward(self, x):
        embeds = self.embedding(x)
        hidden = self.gru(embeds)[1][1]  # select h_n, and select the 2nd layer
        logits = self.linear(hidden)
        return logits


model = ClfGRU(2).cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)


def train():
    model.train()
    loss_ema = 0

    for batch_idx, batch in enumerate(iter(train_iter)):
        inputs = batch.text.t()
        labels = batch.label - 1

        logits = model(inputs)

        loss = F.cross_entropy(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_ema = loss_ema * 0.9 + loss.data.cpu().numpy() * 0.1

        if batch_idx % 200 == 0:
            print('iter: {} | loss_ema: {}'.format(batch_idx, loss_ema))

    scheduler.step()


def evaluate():
    model.eval()
    running_loss = 0
    num_examples = 0
    correct = 0

    for batch_idx, batch in enumerate(iter(test_iter)):
        inputs = batch.text.t()
        labels = batch.label - 1

        logits = model(inputs)

        loss = F.cross_entropy(logits, labels, size_average=False)
        running_loss += loss.data.cpu().numpy()

        pred = logits.max(1)[1]
        correct += pred.eq(labels).sum().data.cpu().numpy()

        num_examples += inputs.shape[0]

    acc = correct / num_examples
    loss = running_loss / num_examples

    return acc, loss


acc, loss = evaluate()
print('test acc: {} \t| test loss: {}\n'.format(acc, loss))
for epoch in range(args.epochs):
    print('Epoch', epoch)
    train()
    acc, loss = evaluate()
    print('test acc: {} \t| test loss: {}\n'.format(acc, loss))


torch.save(model.state_dict(), './snapshots/{}/baseline/model.dict'.format(args.in_dist_dataset))
print('Saved model.')