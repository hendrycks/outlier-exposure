import argparse
import time
import math
import numpy as np
import sklearn.metrics as sk
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

import data
import model

from utils_lm import batchify, get_batch, repackage_hidden

# go through rigamaroo to do ..utils.display_results import show_performance
if __package__ is None:
    import sys
    from os import path

    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    from utils.display_results import show_performance
    from utils.log_sum_exp import log_sum_exp

parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='data/penn/',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (LSTM, QRNN, GRU)')
parser.add_argument('--emsize', type=int, default=400,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=1150,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=3,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=30,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=8000,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=80, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=70,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.4,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--dropouth', type=float, default=0.3,
                    help='dropout for rnn layers (0 = no dropout)')
parser.add_argument('--dropouti', type=float, default=0.65,
                    help='dropout for input embedding layers (0 = no dropout)')
parser.add_argument('--dropoute', type=float, default=0.1,
                    help='dropout to remove words from embedding layer (0 = no dropout)')
parser.add_argument('--wdrop', type=float, default=0.5,
                    help='amount of weight dropout to apply to the RNN hidden to hidden matrix')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--nonmono', type=int, default=5,
                    help='random seed')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
randomhash = ''.join(str(time.time()).split('.'))
parser.add_argument('--save', type=str,  default=randomhash+'.pt',
                    help='path to save the final model')
parser.add_argument('--alpha', type=float, default=2,
                    help='alpha L2 regularization on RNN activation (alpha = 0 means no regularization)')
parser.add_argument('--beta', type=float, default=1,
                    help='beta slowness regularization applied on RNN activiation (beta = 0 means no regularization)')
parser.add_argument('--wdecay', type=float, default=1.2e-6,
                    help='weight decay applied to all weights')
parser.add_argument('--resume', type=str,  default='',
                    help='path of model to resume')
parser.add_argument('--optimizer', type=str,  default='sgd',
                    help='optimizer to use (sgd, adam)')
parser.add_argument('--when', nargs="+", type=int, default=[-1],
                    help='When (which epochs) to divide the learning rate by 10 - accepts multiple')
parser.add_argument('--character_level', action='store_true', help="Use this flag to evaluate character-level models.")
args = parser.parse_args()
args.tied = True

# Set the random seed manually for reproducibility.
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

###############################################################################
# Load data
###############################################################################

def model_save(fn):
    with open(fn, 'wb') as f:
        torch.save([model, criterion, optimizer], f)

def model_load(fn):
    global model, criterion, optimizer
    with open(fn, 'rb') as f:
        model, criterion, optimizer = torch.load(f)

import os
import hashlib
fn = 'corpus.{}.data'.format(hashlib.md5(args.data.encode()).hexdigest())
if os.path.exists(fn):
    print('Loading cached dataset...')
    corpus = torch.load(fn)
else:
    print('Producing dataset...')
    corpus = data.Corpus(args.data)
    torch.save(corpus, fn)

eval_batch_size = 10
test_batch_size = 1  # DON'T CHANGE THIS
train_data = batchify(corpus.train, args.batch_size, args)
val_data = batchify(corpus.valid, eval_batch_size, args)
test_data = batchify(corpus.test, test_batch_size, args)


print('Producing ood datasets...')

answers_corpus = data.OODCorpus('eng_web_tbk/answers/conll/answers_penntrees.dev.conll', corpus.dictionary, char=args.character_level)
answers_data = batchify(answers_corpus.data, test_batch_size, args)

email_corpus = data.OODCorpus('eng_web_tbk/email/conll/email_penntrees.dev.conll', corpus.dictionary, char=args.character_level)
email_data = batchify(email_corpus.data, test_batch_size, args)

newsgroup_corpus = data.OODCorpus('eng_web_tbk/newsgroup/conll/newsgroup_penntrees.dev.conll', corpus.dictionary, char=args.character_level)
newsgroup_data = batchify(newsgroup_corpus.data, test_batch_size, args)

reviews_corpus = data.OODCorpus('eng_web_tbk/reviews/conll/reviews_penntrees.dev.conll', corpus.dictionary, char=args.character_level)
reviews_data = batchify(reviews_corpus.data, test_batch_size, args)

weblog_corpus = data.OODCorpus('eng_web_tbk/weblog/conll/weblog_penntrees.dev.conll', corpus.dictionary, char=args.character_level)
weblog_data = batchify(weblog_corpus.data, test_batch_size, args)


###############################################################################
# Build the model
###############################################################################

from splitcross import SplitCrossEntropyLoss
criterion = None

ntokens = len(corpus.dictionary)
model = model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.dropouth, args.dropouti, args.dropoute, args.wdrop, args.tied)
###
assert args.resume, 'must provide a --resume argument'

print('Resuming model ...')
model_load(args.resume)
optimizer.param_groups[0]['lr'] = args.lr
model.dropouti, model.dropouth, model.dropout, args.dropoute = args.dropouti, args.dropouth, args.dropout, args.dropoute
if args.wdrop:
    from weight_drop import WeightDrop
    for rnn in model.rnns:
        if type(rnn) == WeightDrop: rnn.dropout = args.wdrop
        elif rnn.zoneout > 0: rnn.zoneout = args.wdrop
###
if not criterion:
    splits = []
    if ntokens > 500000:
        # One Billion
        # This produces fairly even matrix mults for the buckets:
        # 0: 11723136, 1: 10854630, 2: 11270961, 3: 11219422
        splits = [4200, 35000, 180000]
    elif ntokens > 75000:
        # WikiText-103
        splits = [2800, 20000, 76000]
    print('Using', splits)
    criterion = SplitCrossEntropyLoss(args.emsize, splits=splits, verbose=False)
###
if args.cuda:
    model = model.cuda()
    criterion = criterion.cuda()
###
params = list(model.parameters()) + list(criterion.parameters())
total_params = sum(x.size()[0] * x.size()[1] if len(x.size()) > 1 else x.size()[0] for x in params if x.size())
print('Args:', args)
print('Model total parameters:', total_params)

###############################################################################
# Eval code
###############################################################################

ood_num_examples = test_data.size(0) // 5
expected_ap = ood_num_examples / (ood_num_examples + test_data.size(0))
recall_level = 0.9


def get_base_rates():
    batch, i = 0, 0
    seq_len = args.bptt
    ntokens = len(corpus.dictionary)
    token_counts = np.zeros(ntokens)
    total_count = 0

    for i in range(0, train_data.size(0), args.bptt):  # Assume OE dataset is larger. It is, because we're using wikitext-2.
        data, targets = get_batch(train_data, i, args, seq_len=seq_len)
        for j in range(targets.numel()):
            token_counts[targets[j].data.cpu().numpy()[0]] += 1
            total_count += 1
        batch += 1

    return token_counts / total_count


print('Getting base rates...')
# base_rates = get_base_rates()
# np.save('./base_rates.npy', base_rates)
base_rates = Variable(torch.from_numpy(np.load('./base_rates.npy').astype(np.float32))).cuda().float().squeeze()  # shit happens
uniform_base_rates = Variable(torch.from_numpy(np.ones(len(corpus.dictionary)).astype(np.float32))).cuda().float().squeeze()
uniform_base_rates /= uniform_base_rates.numel()
print('Done.')


def evaluate(data_source, corpus, batch_size=10, ood=False):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    if args.model == 'QRNN': model.reset()
    loss_accum = 0
    losses = []
    ntokens = len(corpus.dictionary)
    for i in range(0, data_source.size(0) - 1, args.bptt):
        if (i >= ood_num_examples // test_batch_size) and (ood is True):
            break

        hidden = model.init_hidden(batch_size)
        hidden = repackage_hidden(hidden)

        data, targets = get_batch(data_source, i, args, evaluation=True)
        output, hidden = model(data, hidden)
        
        logits = model.decoder(output)
        smaxes = F.softmax(logits - torch.max(logits, dim=1, keepdim=True)[0], dim=1)
        tmp = smaxes[range(targets.size(0)), targets]
        log_prob = torch.log(tmp).mean(0)  # divided by seq len, so this is the negative nats per char
        loss = -log_prob.data.cpu().numpy()[0]
        
        loss_accum += loss
        # losses.append(loss)
        # Experimental!
        # anomaly_score = -torch.max(smaxes, dim=1)[0].mean()  # negative MSP
        anomaly_score = ((smaxes).add(1e-18).log() * uniform_base_rates.unsqueeze(0)).sum(1).mean(0)  # negative KL to uniform
        losses.append(anomaly_score.data.cpu().numpy()[0])
        #

    return loss_accum / (len(data_source) // args.bptt), losses



# Run on test data.
print('\nPTB')
test_loss, test_losses = evaluate(test_data, corpus, test_batch_size)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f} | test bpc {:8.3f}'.format(
    test_loss, math.exp(test_loss), test_loss / math.log(2)))
print('=' * 89)


print('\nAnswers (OOD)')
ood_loss, ood_losses = evaluate(answers_data, answers_corpus, test_batch_size, ood=True)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f} | test bpc {:8.3f}'.format(
    ood_loss, math.exp(ood_loss), ood_loss / math.log(2)))
print('=' * 89)
show_performance(ood_losses, test_losses, expected_ap, recall_level=recall_level)


print('\nEmail (OOD)')
ood_loss, ood_losses = evaluate(email_data, email_corpus, test_batch_size, ood=True)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f} | test bpc {:8.3f}'.format(
    ood_loss, math.exp(ood_loss), ood_loss / math.log(2)))
print('=' * 89)
show_performance(ood_losses, test_losses, expected_ap, recall_level=recall_level)


print('\nNewsgroup (OOD)')
ood_loss, ood_losses = evaluate(newsgroup_data, newsgroup_corpus, test_batch_size, ood=True)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f} | test bpc {:8.3f}'.format(
    ood_loss, math.exp(ood_loss), ood_loss / math.log(2)))
print('=' * 89)
show_performance(ood_losses, test_losses, expected_ap, recall_level=recall_level)


print('\nReviews (OOD)')
ood_loss, ood_losses = evaluate(reviews_data, reviews_corpus, test_batch_size, ood=True)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f} | test bpc {:8.3f}'.format(
    ood_loss, math.exp(ood_loss), ood_loss / math.log(2)))
print('=' * 89)
show_performance(ood_losses, test_losses, expected_ap, recall_level=recall_level)


print('\nWeblog (OOD)')
ood_loss, ood_losses = evaluate(weblog_data, weblog_corpus, test_batch_size, ood=True)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f} | test bpc {:8.3f}'.format(
    ood_loss, math.exp(ood_loss), ood_loss / math.log(2)))
print('=' * 89)
show_performance(ood_losses, test_losses, expected_ap, recall_level=recall_level)
