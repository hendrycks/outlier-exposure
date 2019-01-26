import argparse
import time
import math
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

import data
import model

from utils_lm import batchify, get_batch, repackage_hidden

parser = argparse.ArgumentParser(description='Train with OE using cross-entropy to base rates.')
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
parser.add_argument('--use_OE', type=str)
parser.add_argument('--wikitext_char', action='store_true', help='Load character-level WikiText. Use when in-dist is character-level.')
args = parser.parse_args()
print(args)
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
test_batch_size = 1
train_data = batchify(corpus.train, args.batch_size, args)
val_data = batchify(corpus.valid, eval_batch_size, args)
test_data = batchify(corpus.test, test_batch_size, args)

###############################################################################
# Load OE data
###############################################################################

print('Producing dataset...')
if args.wikitext_char:
    oe_corpus = data.CorpusWikiTextChar('data/wikitext-2', corpus.dictionary)

    oe_dataset = batchify(oe_corpus.train, args.batch_size, args)
    oe_val_dataset = batchify(oe_corpus.valid, eval_batch_size, args)
else:
    oe_corpus = data.Corpus('data/wikitext-2', corpus.dictionary)

    oe_dataset = batchify(oe_corpus.train, args.batch_size, args)
    oe_val_dataset = batchify(oe_corpus.valid, eval_batch_size, args)

###############################################################################
# Build the model
###############################################################################

from splitcross import SplitCrossEntropyLoss
criterion = None

ntokens = len(corpus.dictionary)
model = model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.dropouth, args.dropouti, args.dropoute, args.wdrop, args.tied)
###
if args.resume:
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
# Training code
###############################################################################

def evaluate(data_source, batch_size=10, test=False):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    if args.model == 'QRNN': model.reset()
    total_loss = 0
    total_oe_loss = 0
    num_batches = 0
    ntokens = len(corpus.dictionary)
    for i in range(0, data_source.size(0) - 1, args.bptt):
        data, targets = get_batch(data_source, i, args, evaluation=True)
        data_oe, _ = get_batch(oe_val_dataset, i, args, evaluation=True)

        if len(data.size()) == 1:  # happens for test set?
            data.unsqueeze(-1)
            data_oe.unsqueeze(-1)

        if data.size(0) != data_oe.size(0):
            continue

        bs = test_batch_size if test else eval_batch_size
        hidden = model.init_hidden(2 * bs) 
        hidden = repackage_hidden(hidden)

        output, hidden, rnn_hs, dropped_rnn_hs = model(torch.cat([data, data_oe], dim=1), hidden, return_h=True)
        output, output_oe = torch.chunk(dropped_rnn_hs[-1], dim=1, chunks=2)
        output, output_oe = output.contiguous(), output_oe.contiguous()
        output = output.view(output.size(0)*output.size(1), output.size(2))

        loss = criterion(model.decoder.weight, model.decoder.bias, output, targets).data

        # OE loss
        logits_oe = model.decoder(output_oe)
        smaxes_oe = F.softmax(logits_oe - torch.max(logits_oe, dim=-1, keepdim=True)[0], dim=-1)
        loss_oe = -smaxes_oe.log().mean(-1)
        loss_oe = loss_oe.mean().data
        #

        total_loss += loss
        total_oe_loss += loss_oe
        num_batches += 1
    return total_loss[0] / num_batches, total_oe_loss[0] / num_batches


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


def train(base_rates):
    # Turn on training mode which enables dropout.
    if args.model == 'QRNN': model.reset()
    total_loss = 0
    total_oe_loss = 0
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    batch, i = 0, 0

    # indices for randomizing order of segments
    train_indices = np.arange(train_data.size(0) // args.bptt)
    np.random.shuffle(train_indices)

    oe_indices = np.arange(oe_dataset.size(0) // args.bptt)
    np.random.shuffle(oe_indices)
    #

    seq_len = args.bptt

    br = None

    for i in range(0, train_data.size(0), args.bptt):  # Assume OE dataset is larger. It is, because we're using wikitext-2.

        lr2 = optimizer.param_groups[0]['lr']
        optimizer.param_groups[0]['lr'] = lr2 * seq_len / args.bptt
        model.train()
        data, targets = get_batch(train_data, i, args, seq_len=seq_len)
        data_oe, _ = get_batch(oe_dataset, i, args, seq_len=seq_len)

        if data.size(0) != data_oe.size(0):  # Don't train on this batch if the sequence lengths are different (happens at end of epoch).
            continue

        # We need a new hidden state for each segment, because this makes evaluation easier and more meaningful.
        hidden = model.init_hidden(2 * args.batch_size)
        hidden = repackage_hidden(hidden)

        output, hidden, rnn_hs, dropped_rnn_hs = model(torch.cat([data, data_oe], dim=1), hidden, return_h=True)
        output, output_oe = torch.chunk(dropped_rnn_hs[-1], dim=1, chunks=2)
        output, output_oe = output.contiguous(), output_oe.contiguous()
        output = output.view(output.size(0)*output.size(1), output.size(2))

        raw_loss = criterion(model.decoder.weight, model.decoder.bias, output, targets)


        loss = raw_loss
        # Activiation Regularization
        if args.alpha: loss = loss + sum(args.alpha * dropped_rnn_h.pow(2).mean() for dropped_rnn_h in dropped_rnn_hs[-1:])
        # Temporal Activation Regularization (slowness)
        if args.beta: loss = loss + sum(args.beta * (rnn_h[1:] - rnn_h[:-1]).pow(2).mean() for rnn_h in rnn_hs[-1:])

        # OE loss
        logits_oe = model.decoder(output_oe)
        smaxes_oe = F.softmax(logits_oe - torch.max(logits_oe, dim=-1, keepdim=True)[0], dim=-1)
        br = Variable(torch.FloatTensor(base_rates).unsqueeze(0).unsqueeze(0).expand_as(smaxes_oe)).cuda() if br is None else br
        loss_oe = -(smaxes_oe.log() * br).sum(-1)  # for cross entropy
        loss_oe = loss_oe.mean()  # for ERM
        #

        if args.use_OE == 'yes':
            loss_bp = loss + 0.5 * loss_oe
        else:
            loss_bp = loss

        optimizer.zero_grad()
        loss_bp.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        if args.clip: torch.nn.utils.clip_grad_norm(params, args.clip)
        optimizer.step()

        total_loss += raw_loss.data
        total_oe_loss += loss_oe.data
        optimizer.param_groups[0]['lr'] = lr2
        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss[0] / args.log_interval
            cur_oe_loss = total_oe_loss[0] /args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:05.5f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | oe_loss {:5.2f} | ppl {:8.2f} | bpc {:8.3f}'.format(
                epoch, batch, len(train_data) // args.bptt, optimizer.param_groups[0]['lr'],
                elapsed * 1000 / args.log_interval, cur_loss, cur_oe_loss, math.exp(cur_loss), cur_loss / math.log(2)))
            total_loss = 0
            total_oe_loss = 0
            start_time = time.time()
        ###
        batch += 1

# Loop over epochs.
lr = args.lr
best_val_loss = []
stored_loss = 100000000

# At any point you can hit Ctrl + C to break out of training early.
try:
    optimizer = None
    # Ensure the optimizer is optimizing params, which includes both the model's weights as well as the criterion's weight (i.e. Adaptive Softmax)
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(params, lr=args.lr, weight_decay=args.wdecay)
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.wdecay)

    base_rates = get_base_rates()
    
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        train(base_rates)
        if 't0' in optimizer.param_groups[0]:
            tmp = {}
            for prm in model.parameters():
                tmp[prm] = prm.data.clone()
                prm.data = optimizer.state[prm]['ax'].clone()

            val_loss2, val_oe_loss = evaluate(val_data)
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | val oe_loss {:5.2f} | '
                'valid ppl {:8.2f} | valid bpc {:8.3f}'.format(
              epoch, (time.time() - epoch_start_time), val_loss, val_oe_loss, math.exp(val_loss), val_loss / math.log(2)))
            print('-' * 89)

            if val_loss2 < stored_loss:
                model_save(args.save)
                print('Saving Averaged!')
                stored_loss = val_loss2

            for prm in model.parameters():
                prm.data = tmp[prm].clone()

        else:
            val_loss, val_oe_loss = evaluate(val_data, eval_batch_size)
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | val oe_loss {:5.2f} | '
                'valid ppl {:8.2f} | valid bpc {:8.3f}'.format(
              epoch, (time.time() - epoch_start_time), val_loss, val_oe_loss, math.exp(val_loss), val_loss / math.log(2)))
            print('-' * 89)

            if val_loss < stored_loss:
                model_save(args.save)
                print('Saving model (new best validation)')
                stored_loss = val_loss

            if args.optimizer == 'sgd' and 't0' not in optimizer.param_groups[0] and (len(best_val_loss)>args.nonmono and val_loss > min(best_val_loss[:-args.nonmono])):
                print('Switching to ASGD')
                optimizer = torch.optim.ASGD(model.parameters(), lr=args.lr, t0=0, lambd=0., weight_decay=args.wdecay)

            if epoch in args.when:
                print('Saving model before learning rate decreased')
                model_save('{}.e{}'.format(args.save, epoch))
                print('Dividing learning rate by 10')
                optimizer.param_groups[0]['lr'] /= 10.

            best_val_loss.append(val_loss)

except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Load the best saved model.
model_load(args.save)

# Run on test data.
test_loss, val_oe_loss = evaluate(test_data, test_batch_size)
print('=' * 89)
print('| End of training | test loss {:5.2f} | val oe_loss {:5.2f} | test ppl {:8.2f} | test bpc {:8.3f}'.format(
    test_loss, val_oe_loss, math.exp(test_loss), test_loss / math.log(2)))
print('=' * 89)
