# -*- coding: utf-8 -*-
import numpy as np
import sys
import os
import pickle
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as trn
import torchvision.datasets as dset
import torch.nn.functional as F
from torch.autograd import Variable as V
from skimage.filters import gaussian as gblur

# go through rigamaroo to do ...utils.display_results import show_performance
if __package__ is None:
    import sys
    from os import path

    sys.path.append(path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))
    from utils.display_results import show_performance
    import utils.svhn_loader as svhn
    import utils.lsun_loader as lsun_loader
    from utils.cifar_resnet import WideResNet

parser = argparse.ArgumentParser(description='Evaluates Anomaly Detectors with a WideResNet on SVHN',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# Optimization options

parser.add_argument('--batch_size', '-b', type=int, default=100, help='Batch size.')
parser.add_argument('--test_bs', type=int, default=200)
# Checkpoints
parser.add_argument('--load', '-l', type=str, default='./snapshots', help='Checkpoint path to resume / test.')
# Architecture
parser.add_argument('--layers', default=16, type=int, help='total number of layers (default: 28)')
parser.add_argument('--widen-factor', default=4, type=int, help='widen factor (default: 10)')
parser.add_argument('--droprate', default=0.4, type=float, help='dropout probability (default: 0.0)')
# Acceleration
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
parser.add_argument('--prefetch', type=int, default=2, help='Pre-fetching threads.')
args = parser.parse_args()

# torch.manual_seed(1)
# np.random.seed(1)

test_data = svhn.SVHN(root='/share/data/vision-greg/svhn/', split="test", transform=trn.ToTensor(), download=False)
num_labels = 10

test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.test_bs, shuffle=False,
                                          num_workers=args.prefetch, pin_memory=True)

# Create model
net = WideResNet(args.layers, num_labels, args.widen_factor, dropRate=args.droprate)

# Restore model
start_epoch = 0
if args.load != '':
    for i in range(300 - 1, -1, -1):
        model_name = os.path.join(args.load, 'svhn_model_epoch' + str(i) + '.pytorch')
        if os.path.isfile(model_name):
            net.load_state_dict(torch.load(model_name))
            print('Model restored! Epoch:', i)
            start_epoch = i + 1
            break
    if start_epoch == 0:
        assert False, "could not resume"

for p in net.parameters():
    p.volatile = True

if args.ngpu > 1:
    net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))

if args.ngpu > 0:
    net.cuda()
    # torch.cuda.manual_seed(1)

net.eval()
cudnn.benchmark = True  # fire on all cylinders


# /////////////// Detection Prelims ///////////////

ood_num_examples = test_data.test_data.shape[0] // 5
expected_ap = ood_num_examples / (ood_num_examples + test_data.test_data.shape[0])

concat = lambda x: np.concatenate(x, axis=0)
to_np = lambda x: x.data.cpu().numpy()


def get_anomaly_scores(loader, in_dist=False):
    _max_prob = []
    _right_max_prob = []
    _wrong_max_prob = []

    for batch_idx, (data, target) in enumerate(loader):
        if batch_idx >= ood_num_examples // args.test_bs and in_dist is False:
            break

        data = V(data.cuda(), volatile=True)

        output = net(data)
        smax = to_np(F.softmax(output, dim=1))

        _max_prob.append(-np.max(smax, axis=1))

        if in_dist:
            preds = np.argmax(smax, axis=1)
            targets = target.numpy().squeeze()
            right_indices = preds == targets
            wrong_indices = np.invert(right_indices)

            _right_max_prob.append(-np.max(smax[right_indices], axis=1))
            _wrong_max_prob.append(-np.max(smax[wrong_indices], axis=1))

    if in_dist:
        return concat(_max_prob).copy(), concat(_right_max_prob).copy(), concat(_wrong_max_prob).copy()
    else:
        return concat(_max_prob)[:ood_num_examples].copy()


in_max_prob, right_max_prob, wrong_max_prob = get_anomaly_scores(test_loader, in_dist=True)

num_right = len(right_max_prob)
num_wrong = len(wrong_max_prob)

recall_level = 0.99

# /////////////// End Detection Prelims ///////////////

print('Using SVHN as typical data')

# /////////////// Error Detection ///////////////

print('\n\nError Detection')
show_performance(wrong_max_prob, right_max_prob, method_name='Baseline', recall_level=recall_level)

# /////////////// Gaussian Noise ///////////////

dummy_targets = torch.ones(ood_num_examples)
ood_data = torch.from_numpy(
    np.clip(np.random.normal(size=(ood_num_examples, 3, 32, 32), loc=0.5, scale=0.5).astype(np.float32), 0, 1))
ood_data = torch.utils.data.TensorDataset(ood_data, dummy_targets)
ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.batch_size, shuffle=True)

out_max_prob = get_anomaly_scores(ood_loader)

print('\n\nGaussian Noise (mu, sigma = 0.5) Detection')
show_performance(out_max_prob, in_max_prob, method_name='Baseline', recall_level=recall_level)

# /////////////// Bernoulli Noise ///////////////

dummy_targets = torch.ones(ood_num_examples)
ood_data = torch.from_numpy(np.random.binomial(n=1, p=0.5, size=(ood_num_examples, 3, 32, 32)).astype(np.float32))
ood_data = torch.utils.data.TensorDataset(ood_data, dummy_targets)
ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.batch_size, shuffle=True)

out_max_prob = get_anomaly_scores(ood_loader)

print('\n\nBernoulli Noise Detection')
show_performance(out_max_prob, in_max_prob, method_name='Baseline', recall_level=recall_level)

# /////////////// Blob ///////////////

ood_data = np.float32(np.random.binomial(n=1, p=0.7, size=(ood_num_examples, 32, 32, 3)))
for i in range(ood_num_examples):
    ood_data[i] = gblur(ood_data[i], sigma=1.5, multichannel=False)
    ood_data[i][ood_data[i] < 0.75] = 0.0

dummy_targets = torch.ones(ood_num_examples)
ood_data = torch.from_numpy(ood_data.transpose((0,3,1,2)))
ood_data = torch.utils.data.TensorDataset(ood_data, dummy_targets)
ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.batch_size, shuffle=True,
                                         num_workers=args.prefetch, pin_memory=True)

out_max_prob = get_anomaly_scores(ood_loader)

print('\n\nBlob Detection')
show_performance(out_max_prob, in_max_prob, method_name='Baseline', recall_level=recall_level)

# validation
# # /////////////// Maximum Input ///////////////
#
# dummy_targets = torch.ones(ood_num_examples)
# ood_data = torch.utils.data.TensorDataset(torch.ones((ood_num_examples, 3, 32, 32)), dummy_targets)
# ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.batch_size, shuffle=True)
#
# out_max_prob = get_anomaly_scores(ood_loader)
#
# print('\n\nMaximum Input Detection')
# show_performance(out_max_prob, in_max_prob, method_name='Baseline', recall_level=recall_level)

# # /////////////// Average Images ///////////////
#
# shuffle_indices = np.arange(test_data.test_data.shape[0])
# np.random.shuffle(shuffle_indices)
#
# dummy_targets = torch.ones(test_data.test_data.shape[0])
# ood_data = torch.from_numpy((np.float32(test_data.test_data / 2. + test_data.test_data[shuffle_indices] / 2.) / 255))
# ood_data = torch.utils.data.TensorDataset(ood_data, dummy_targets)
# ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.batch_size, shuffle=True,
#                                          num_workers=args.prefetch, pin_memory=True)
#
# out_max_prob = get_anomaly_scores(ood_loader)
#
# print('\n\nAverage of Random Image Pair Detection')
# show_performance(out_max_prob, in_max_prob, method_name='Baseline', recall_level=recall_level)

# /////////////// Icons-50 ///////////////

ood_data = dset.ImageFolder('/share/data/vision-greg/DistortedImageNet/Icons-50',
                            transform=trn.Compose([trn.Resize((32, 32)), trn.ToTensor()]))

filtered_imgs = []
for img in ood_data.imgs:
    if 'numbers' not in img[0]:     # img[0] is image name
        filtered_imgs.append(img)
ood_data.imgs = filtered_imgs

ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True)

out_max_prob = get_anomaly_scores(ood_loader)

print('\n\nIcons-50 Detection')
show_performance(out_max_prob, in_max_prob, expected_ap, method_name='Baseline', recall_level=recall_level)

# /////////////// Textures ///////////////

ood_data = dset.ImageFolder(root="/share/data/lang/users/dan/datasets/dtd/images",
                            transform=trn.Compose([trn.Resize(32), trn.CenterCrop(32),
                                                   trn.ToTensor()]))
ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.batch_size, shuffle=True,
                                         num_workers=args.prefetch, pin_memory=True)

out_max_prob = get_anomaly_scores(ood_loader)

print('\n\nTexture Detection')
show_performance(out_max_prob, in_max_prob, method_name='Baseline', recall_level=recall_level)

# /////////////// Places365 ///////////////

ood_data = dset.ImageFolder(root="/share/data/lang/users/dan/datasets/places365/test_subset",
                            transform=trn.Compose([trn.Resize(32), trn.CenterCrop(32),
                                                   trn.ToTensor()]))
ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.batch_size, shuffle=True,
                                         num_workers=args.prefetch, pin_memory=True)

out_max_prob = get_anomaly_scores(ood_loader)

print('\n\nPlaces365 Detection')
show_performance(out_max_prob, in_max_prob, method_name='Baseline', recall_level=recall_level)

# /////////////// LSUN ///////////////

ood_data = lsun_loader.LSUN("/share/data/lang/users/dan/datasets/LSUN/lsun-master/data", classes='test',
                            transform=trn.Compose([trn.Resize(32), trn.CenterCrop(32),
                                                   trn.ToTensor()]))

ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.batch_size, shuffle=True,
                                         num_workers=args.prefetch, pin_memory=True)

out_max_prob = get_anomaly_scores(ood_loader)

print('\n\nLSUN Detection')
show_performance(out_max_prob, in_max_prob, method_name='Baseline', recall_level=recall_level)

# /////////////// CIFAR data ///////////////

ood_data = dset.CIFAR10('~/cifar_data', train=False, transform=trn.ToTensor(), download=False)
ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True,
                                         num_workers=args.prefetch, pin_memory=True)

out_max_prob = get_anomaly_scores(ood_loader)

print('\n\nCIFAR-10 Detection')
show_performance(out_max_prob, in_max_prob, method_name='Baseline', recall_level=recall_level)

# /////////////// Street View Characters data ///////////////

ood_data = dset.ImageFolder(root="/share/data/lang/users/dan/datasets/StreetLetters",
                            transform=trn.Compose([trn.Resize(32), trn.CenterCrop(32),
                                                   trn.ToTensor()]))
ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.batch_size, shuffle=True,
                                         num_workers=args.prefetch, pin_memory=True)

out_max_prob = get_anomaly_scores(ood_loader)

print('\n\nStreet View Characters Detection')
show_performance(out_max_prob, in_max_prob, method_name='Baseline', recall_level=recall_level)

# validation
# # /////////////// Mirrored SVHN digits ///////////////
#
# idxs = test_data.test_labels
# vert_idxs = np.squeeze(np.logical_and(idxs != 3, np.logical_and(idxs != 0, np.logical_and(idxs != 1, idxs != 8))))
# vert_digits = test_data.test_data[vert_idxs][:, :, ::-1, :]
#
# horiz_idxs = np.squeeze(np.logical_and(idxs != 0, np.logical_and(idxs != 1, idxs != 8)))
# horiz_digits = test_data.test_data[horiz_idxs][:, :, :, ::-1]
#
# flipped_digits = concat((vert_digits, horiz_digits))
#
# dummy_targets = torch.ones(flipped_digits.shape[0])
# ood_data = torch.from_numpy(flipped_digits.astype(np.float32) / 255)
# ood_data = torch.utils.data.TensorDataset(ood_data, dummy_targets)
# ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.batch_size, shuffle=True,
#                                          num_workers=args.prefetch)
#
# out_max_prob = get_anomaly_scores(ood_loader)
#
# print('\n\nMirrored SVHN Digit Detection')
# show_performance(out_max_prob, in_max_prob, method_name='Baseline', recall_level=recall_level)
