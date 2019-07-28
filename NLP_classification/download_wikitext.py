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

parser = argparse.ArgumentParser(description='SST OE',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
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


# ============================ SST ============================ #
# set up fields
TEXT_sst = data.Field(pad_first=True)
LABEL_sst = data.Field(sequential=False)

# make splits for data
train_sst, val_sst, test_sst = datasets.SST.splits(
    TEXT_sst, LABEL_sst, fine_grained=False, train_subtrees=False,
    filter_pred=lambda ex: ex.label != 'neutral')

# build vocab
TEXT_sst.build_vocab(train_sst, max_size=10000)
LABEL_sst.build_vocab(train_sst, max_size=10000)
print('vocab length (including special tokens):', len(TEXT_sst.vocab))

# create our own iterator, avoiding the calls to build_vocab in SST.iters
train_iter_sst, val_iter_sst, test_iter_sst = data.BucketIterator.splits(
    (train_sst, val_sst, test_sst), batch_size=args.batch_size, repeat=False)
# ============================ SST ============================ #

# ============================ WikiText-2 ============================ #

# set up fields
TEXT_wtxt = data.Field(pad_first=True, lower=True)

# make splits for data
train_OE, val_OE, test_OE = datasets.WikiText2.splits(TEXT_wtxt)

# build vocab
TEXT_wtxt.build_vocab(train_sst.text, max_size=10000)
print('vocab length (including special tokens):', len(TEXT_wtxt.vocab))

# create our own iterator, avoiding the calls to build_vocab in SST.iters
train_iter_oe, val_iter_oe, test_iter_oe = data.BPTTIterator.splits(
    (train_OE, val_OE, test_OE), batch_size=args.batch_size, bptt_len=15, repeat=False)

# ============================ WikiText-2 ============================ #

# ============================ WikiText-103 ============================ #

# set up fields
TEXT_wtxt = data.Field(pad_first=True, lower=True)

# make splits for data
train_OE, val_OE, test_OE = datasets.WikiText103.splits(TEXT_wtxt)

# build vocab
TEXT_wtxt.build_vocab(train_sst.text, max_size=10000)
print('vocab length (including special tokens):', len(TEXT_wtxt.vocab))

# create our own iterator, avoiding the calls to build_vocab in SST.iters
train_iter_oe, val_iter_oe, test_iter_oe = data.BPTTIterator.splits(
    (train_OE, val_OE, test_OE), batch_size=args.batch_size, bptt_len=15, repeat=False)

# ============================ WikiText-103 ============================ #

exit()