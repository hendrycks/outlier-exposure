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
import spacy
import re

import tqdm



np.random.seed(1)

parser = argparse.ArgumentParser(description='SST OE',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--batch_size', '-b', type=int, default=64, help='Batch size.')
args = parser.parse_args()


torch.set_grad_enabled(False)
cudnn.benchmark = True  # fire on all cylinders


# go through rigamaroo to do ..utils.display_results import show_performance
if __package__ is None:
    import sys
    from os import path

    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    from utils.display_results import get_performance



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

ood_num_examples = len(test_iter_sst.dataset) // 5
expected_ap = ood_num_examples / (ood_num_examples + len(test_iter_sst.dataset))
recall_level = 0.9

# ============================ IMDB ============================ #

# set up fields
TEXT_imdb = data.Field(pad_first=True, lower=True)
LABEL_imdb = data.Field(sequential=False)

# make splits for data
train_imdb, test_imdb = datasets.IMDB.splits(TEXT_imdb, LABEL_imdb)

# build vocab
TEXT_imdb.build_vocab(train_sst.text, max_size=10000)
LABEL_imdb.build_vocab(train_imdb, max_size=10000)
print('vocab length (including special tokens):', len(TEXT_imdb.vocab))

# make iterators
train_iter_imdb, test_iter_imdb = data.BucketIterator.splits(
    (train_imdb, test_imdb), batch_size=args.batch_size, repeat=False)

# ============================ IMDB ============================ #

# ============================ SNLI ============================ #

# set up fields
TEXT_snli = data.Field(pad_first=True, lower=True)
LABEL_snli = data.Field(sequential=False)

# make splits for data
train_snli, val_snli, test_snli = datasets.SNLI.splits(TEXT_snli, LABEL_snli)

# build vocab
TEXT_snli.build_vocab(train_sst.text, max_size=10000)
LABEL_snli.build_vocab(train_snli, max_size=10000)
print('vocab length (including special tokens):', len(TEXT_snli.vocab))

# make iterators
train_iter_snli, val_iter_snli, test_iter_snli = data.BucketIterator.splits(
    (train_snli, val_snli, test_snli), batch_size=args.batch_size, repeat=False)

# ============================ SNLI ============================ #

# ============================ Multi30K ============================ #
TEXT_m30k = data.Field(pad_first=True, lower=True)

m30k_data = data.TabularDataset(path='./.data/multi30k/train.txt',
                                  format='csv',
                                  fields=[('text', TEXT_m30k)])

TEXT_m30k.build_vocab(train_sst.text, max_size=10000)
print('vocab length (including special tokens):', len(TEXT_m30k.vocab))

train_iter_m30k = data.BucketIterator(m30k_data, batch_size=args.batch_size, repeat=False)
# ============================ Multi30K ============================ #

# ============================ WMT16 ============================ #
TEXT_wmt16 = data.Field(pad_first=True, lower=True)

wmt16_data = data.TabularDataset(path='./.data/wmt16/wmt16_sentences',
                                  format='csv',
                                  fields=[('text', TEXT_wmt16)])

TEXT_wmt16.build_vocab(train_sst.text, max_size=10000)
print('vocab length (including special tokens):', len(TEXT_wmt16.vocab))

train_iter_wmt16 = data.BucketIterator(wmt16_data, batch_size=args.batch_size, repeat=False)
# ============================ WMT16 ============================ #

# ============================ English Web Treebank (Answers) ============================ #

TEXT_answers = data.Field(pad_first=True, lower=True)

treebank_path = './.data/eng_web_tbk/answers/conll/answers_penntrees.dev.conll'

train_answers = datasets.SequenceTaggingDataset(path=treebank_path, fields=((None, None), ('text', TEXT_answers)))

TEXT_answers.build_vocab(train_sst.text, max_size=10000)
print('vocab length (including special tokens):', len(TEXT_answers.vocab))

# make iterators
train_iter_answers = data.BucketIterator.splits(
    (train_answers,), batch_size=args.batch_size, repeat=False)[0]

# ============================ English Web Treebank (Answers) ============================ #

# ============================ English Web Treebank (Email) ============================ #

TEXT_email = data.Field(pad_first=True, lower=True)

treebank_path = './.data/eng_web_tbk/email/conll/email_penntrees.dev.conll'

train_email = datasets.SequenceTaggingDataset(path=treebank_path, fields=((None, None), ('text', TEXT_email)))

TEXT_email.build_vocab(train_sst.text, max_size=10000)
print('vocab length (including special tokens):', len(TEXT_email.vocab))

# make iterators
train_iter_email = data.BucketIterator.splits(
    (train_email,), batch_size=args.batch_size, repeat=False)[0]

# ============================ English Web Treebank (Email) ============================ #

# ============================ English Web Treebank (Newsgroup) ============================ #

TEXT_newsgroup = data.Field(pad_first=True, lower=True)

treebank_path = './.data/eng_web_tbk/newsgroup/conll/newsgroup_penntrees.dev.conll'

train_newsgroup = datasets.SequenceTaggingDataset(path=treebank_path, fields=((None, None), ('text', TEXT_newsgroup)))

TEXT_newsgroup.build_vocab(train_sst.text, max_size=10000)
print('vocab length (including special tokens):', len(TEXT_newsgroup.vocab))

# make iterators
train_iter_newsgroup = data.BucketIterator.splits(
    (train_newsgroup,), batch_size=args.batch_size, repeat=False)[0]

# ============================ English Web Treebank (Newsgroup) ============================ #

# ============================ English Web Treebank (Reviews) ============================ #

TEXT_reviews = data.Field(pad_first=True, lower=True)

treebank_path = './.data/eng_web_tbk/reviews/conll/reviews_penntrees.dev.conll'

train_reviews = datasets.SequenceTaggingDataset(path=treebank_path, fields=((None, None), ('text', TEXT_reviews)))

TEXT_reviews.build_vocab(train_sst.text, max_size=10000)
print('vocab length (including special tokens):', len(TEXT_reviews.vocab))

# make iterators
train_iter_reviews = data.BucketIterator.splits(
    (train_reviews,), batch_size=args.batch_size, repeat=False)[0]

# ============================ English Web Treebank (Reviews) ============================ #

# ============================ English Web Treebank (Weblog) ============================ #

TEXT_weblog = data.Field(pad_first=True, lower=True)

treebank_path = './.data/eng_web_tbk/weblog/conll/weblog_penntrees.dev.conll'

train_weblog = datasets.SequenceTaggingDataset(path=treebank_path, fields=((None, None), ('text', TEXT_weblog)))

TEXT_weblog.build_vocab(train_sst.text, max_size=10000)
print('vocab length (including special tokens):', len(TEXT_weblog.vocab))

# make iterators
train_iter_weblog = data.BucketIterator.splits(
    (train_weblog,), batch_size=args.batch_size, repeat=False)[0]

# ============================ English Web Treebank (Weblog) ============================ #

# ============================ Yelp Reviews ============================ #
TEXT_yelp = data.Field(pad_first=True, lower=True)

yelp_data = data.TabularDataset(path='./.data/yelp_review_full_csv/test.csv',
                                  format='csv',
                                  fields=[(None, None), ('text', TEXT_yelp)])

TEXT_yelp.build_vocab(train_sst.text, max_size=10000)
print('vocab length (including special tokens):', len(TEXT_yelp.vocab))

train_iter_yelp = data.BucketIterator(yelp_data, batch_size=args.batch_size, repeat=False)
# ============================ Yelp Reviews ============================ #




class ClfGRU(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(len(TEXT_sst.vocab), 50, padding_idx=1)
        self.gru = nn.GRU(input_size=50, hidden_size=128, num_layers=2, bias=True, batch_first=True, bidirectional=False)
        self.linear = nn.Linear(128, num_classes)
        self.num_classes = num_classes

    def forward(self, x):
        embeds = self.embedding(x)
        hidden = self.gru(embeds)[1][1]  # select h_n, and select the 2nd layer
        logits = self.linear(hidden)
        return logits



model = ClfGRU(2).cuda()
model.load_state_dict(torch.load('./snapshots/sst/baseline/model.dict'))
print('\nLoaded model.\n')



def get_scores(dataset_iterator, ood=False, snli=False):
    model.eval()

    outlier_scores = []

    for batch_idx, batch in enumerate(iter(dataset_iterator)):
        if ood and (batch_idx * args.batch_size > ood_num_examples):
            break

        if snli:
            inputs = batch.hypothesis.t()
        else:
            inputs = batch.text.t()

        logits = model(inputs)
        smax = F.softmax(logits - torch.max(logits, dim=1, keepdim=True)[0], dim=1)
        msp = -1 * torch.max(smax, dim=1)[0]

        # ce_to_unif = F.log_softmax(logits - torch.max(logits, dim=1, keepdim=True)[0], dim=1).mean(1)  # negative cross entropy
        # test = (F.softmax(logits - torch.max(logits, dim=1, keepdim=True)[0], dim=1) * (1 / torch.FloatTensor([logits.size(1)]).cuda().mean()).log()).sum(1)
        # test = -1 * (F.log_softmax(logits - torch.max(logits, dim=1, keepdim=True)[0], dim=1) * smax).sum(1)

        outlier_scores.extend(list(msp.data.cpu().numpy()))

    return outlier_scores



# ============================ OE ============================ #

test_scores = get_scores(test_iter_sst)

titles = ['SNLI', 'IMDB', 'Multi30K', 'WMT16', 'English Web Treebank (Answers)',
          'English Web Treebank (Email)', 'English Web Treebank (Newsgroup)',
          'English Web Treebank (Reviews)', 'English Web Treebank (Weblog)',
          'Yelp Reviews']

iterators = [test_iter_snli, test_iter_imdb, train_iter_m30k, train_iter_wmt16, train_iter_answers,
             train_iter_email, train_iter_newsgroup, train_iter_reviews, train_iter_weblog,
             train_iter_yelp]


mean_fprs = []
mean_aurocs = []
mean_auprs = []

for i in range(len(titles)):
    title = titles[i]
    iterator = iterators[i]

    print('\n{}'.format(title))
    fprs, aurocs, auprs = [], [], []
    for i in range(10):
        ood_scores = get_scores(iterator, ood=True, snli=True) if 'SNLI' in title else get_scores(iterator, ood=True)
        fpr, auroc, aupr = get_performance(ood_scores, test_scores, expected_ap, recall_level=recall_level)
        fprs.append(fpr)
        aurocs.append(auroc)
        auprs.append(aupr)

    print('FPR{:d}:\t\t\t{:.4f} ({:.4f})'.format(int(100 * recall_level), np.mean(fprs), np.std(fprs)))
    print('AUROC:\t\t\t{:.4f} ({:.4f})'.format(np.mean(aurocs), np.std(aurocs)))
    print('AUPR:\t\t\t{:.4f} ({:.4f})'.format(np.mean(auprs), np.std(auprs)))

    mean_fprs.append(np.mean(fprs))
    mean_aurocs.append(np.mean(aurocs))
    mean_auprs.append(np.mean(auprs))

print()
print('OOD dataset mean FPR: {:.4f}'.format(np.mean(mean_fprs)))
print('OOD dataset mean AUROC: {:.4f}'.format(np.mean(mean_aurocs)))
print('OOD dataset mean AUPR: {:.4f}'.format(np.mean(mean_auprs)))