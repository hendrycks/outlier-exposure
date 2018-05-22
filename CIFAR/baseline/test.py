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
    from utils.log_sum_exp import log_sum_exp

parser = argparse.ArgumentParser(description='Evaluates the Baseline with a WRN on CIFAR',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# Positional arguments
parser.add_argument('dataset', type=str, choices=['cifar10', 'cifar100'],
                    help='Choose between CIFAR-10, CIFAR-100.')
# Optimization options
parser.add_argument('--batch_size', '-b', type=int, default=200, help='Batch size.')
parser.add_argument('--test_bs', type=int, default=200)
parser.add_argument('--layers', default=40, type=int, help='total number of layers (default: 28)')
parser.add_argument('--widen-factor', default=4, type=int, help='widen factor (default: 10)')
parser.add_argument('--droprate', default=0.3, type=float, help='dropout probability (default: 0.0)')
parser.add_argument('--load', '-l', type=str, default='./snapshots', help='Checkpoint path to resume / test.')
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
parser.add_argument('--prefetch', type=int, default=2, help='Pre-fetching threads.')
args = parser.parse_args()

# torch.manual_seed(1)
# np.random.seed(1)

mean = [x / 255 for x in [125.3, 123.0, 113.9]]
std = [x / 255 for x in [63.0, 62.1, 66.7]]

test_transform = trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)])

if args.dataset == 'cifar10':
    test_data = dset.CIFAR10('/share/data/vision-greg/cifarpy', train=False, transform=test_transform, download=False)
    num_classes = 10
else:
    test_data = dset.CIFAR100('/share/data/vision-greg/cifarpy', train=False, transform=test_transform, download=False)
    num_classes = 100

test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.test_bs, shuffle=False,
                                          num_workers=args.prefetch, pin_memory=True)

# Create model
net = WideResNet(args.layers, num_classes, args.widen_factor, dropRate=args.droprate)

# Restore model
start_epoch = 0
if args.load != '':
    for i in range(300 - 1, -1, -1):
        model_name = os.path.join(args.load, args.dataset + '_model_epoch' + str(i) + '.pytorch')
        if os.path.isfile(model_name):
            net.load_state_dict(torch.load(model_name))
            print('Model restored! Epoch:', i)
            start_epoch = i + 1
            break
    if start_epoch == 0:
        assert False, "could not resume"

net.eval()

for p in net.parameters():
    p.volatile = True

if args.ngpu > 1:
    net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))

if args.ngpu > 0:
    net.cuda()
    # torch.cuda.manual_seed(1)

cudnn.benchmark = True  # fire on all cylinders

# /////////////// Detection Prelims ///////////////

ood_num_examples = test_data.test_data.shape[0] // 5
expected_ap = ood_num_examples / (ood_num_examples + test_data.test_data.shape[0])

concat = lambda x: np.concatenate(x, axis=0)
to_np = lambda x: x.data.cpu().numpy()


def get_anomaly_scores(loader, in_dist=False):
    _score = []
    _right_score = []
    _wrong_score = []

    for batch_idx, (data, target) in enumerate(loader):
        if batch_idx >= ood_num_examples // args.test_bs and in_dist is False:
            break

        data = V(data.cuda(), volatile=True)

        output = net(data)
        smax = to_np(F.softmax(output, dim=1))

        _score.append(-np.max(smax, axis=1))

        if in_dist:
            preds = np.argmax(smax, axis=1)
            targets = target.numpy().squeeze()
            right_indices = preds == targets
            wrong_indices = np.invert(right_indices)

            _right_score.append(-np.max(smax[right_indices], axis=1))
            _wrong_score.append(-np.max(smax[wrong_indices], axis=1))

    if in_dist:
        return concat(_score).copy(), concat(_right_score).copy(), concat(_wrong_score).copy()
    else:
        return concat(_score)[:ood_num_examples].copy()


in_max_prob, right_max_prob, wrong_max_prob = get_anomaly_scores(test_loader, in_dist=True)

num_right = len(right_max_prob)
num_wrong = len(wrong_max_prob)

# /////////////// End Detection Prelims ///////////////

print('\nUsing CIFAR-10 as typical data') if num_classes == 10 else print('\nUsing CIFAR-100 as typical data')

# /////////////// Error Detection ///////////////

print('\n\nError Detection')
show_performance(wrong_max_prob, right_max_prob, method_name='Baseline')

# /////////////// Gaussian Noise ///////////////

dummy_targets = torch.ones(test_data.test_data.shape[0])
ood_data = torch.from_numpy(np.float32(np.clip(
    np.random.normal(size=test_data.test_data.shape, scale=0.5).transpose((0, 3, 1, 2)), -1, 1)))
ood_data = torch.utils.data.TensorDataset(ood_data, dummy_targets)
ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.batch_size, shuffle=True,
                                         num_workers=args.prefetch, pin_memory=True)

out_max_prob = get_anomaly_scores(ood_loader)

print('\n\nGaussian Noise (sigma = 0.5) Detection')
show_performance(out_max_prob, in_max_prob, method_name='Baseline')

# /////////////// Rademacher Noise ///////////////

dummy_targets = torch.ones(ood_num_examples)
ood_data = torch.from_numpy(np.random.binomial(n=1, p=0.5, size=(ood_num_examples, 3, 32, 32)).astype(np.float32))*2-1
ood_data = torch.utils.data.TensorDataset(ood_data, dummy_targets)
ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.batch_size, shuffle=False)

out_max_prob = get_anomaly_scores(ood_loader)

print('\n\nRademacher Noise Detection')
show_performance(out_max_prob, in_max_prob, method_name='Baseline')

# /////////////// Blob ///////////////

ood_data = np.float32(np.random.binomial(n=1, p=0.7, size=(ood_num_examples, 32, 32, 3)))
for i in range(ood_num_examples):
    ood_data[i] = gblur(ood_data[i], sigma=1.5, multichannel=False)
    ood_data[i][ood_data[i] < 0.75] = 0.0

dummy_targets = torch.ones(ood_num_examples)
ood_data = torch.from_numpy(ood_data.transpose((0,3,1,2)))*2-1
ood_data = torch.utils.data.TensorDataset(ood_data, dummy_targets)
ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.batch_size, shuffle=True,
                                         num_workers=args.prefetch, pin_memory=True)

out_max_prob = get_anomaly_scores(ood_loader)

print('\n\nBlob Detection')
show_performance(out_max_prob, in_max_prob, method_name='Baseline')


# /////////////// Textures ///////////////

ood_data = dset.ImageFolder(root="/share/data/lang/users/dan/datasets/dtd/images",
                            transform=trn.Compose([trn.Resize(32), trn.CenterCrop(32),
                                                   trn.ToTensor(), trn.Normalize(mean, std)]))
ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.batch_size, shuffle=True,
                                         num_workers=args.prefetch, pin_memory=True)

out_max_prob = get_anomaly_scores(ood_loader)

print('\n\nTexture Detection')
show_performance(out_max_prob, in_max_prob, method_name='Baseline')

# /////////////// SVHN ///////////////

ood_data = svhn.SVHN(root='/share/data/vision-greg/svhn/', split="test",
                     transform=trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)]), download=False)
ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.batch_size, shuffle=True,
                                         num_workers=args.prefetch, pin_memory=True)

out_max_prob = get_anomaly_scores(ood_loader)

print('\n\nSVHN Detection')
show_performance(out_max_prob, in_max_prob, method_name='Baseline')

# /////////////// Places365 ///////////////

ood_data = dset.ImageFolder(root="/share/data/lang/users/dan/datasets/places365/test_subset",
                            transform=trn.Compose([trn.Resize(32), trn.CenterCrop(32),
                                                   trn.ToTensor(), trn.Normalize(mean, std)]))
ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.batch_size, shuffle=True,
                                         num_workers=args.prefetch, pin_memory=True)

out_max_prob = get_anomaly_scores(ood_loader)

print('\n\nPlaces365 Detection')
show_performance(out_max_prob, in_max_prob, method_name='Baseline')

# /////////////// LSUN ///////////////

ood_data = lsun_loader.LSUN("/share/data/lang/users/dan/datasets/LSUN/lsun-master/data", classes='test',
                            transform=trn.Compose([trn.Resize(32), trn.CenterCrop(32),
                                                   trn.ToTensor(), trn.Normalize(mean, std)]))
ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.batch_size, shuffle=True,
                                         num_workers=args.prefetch, pin_memory=True)

out_max_prob = get_anomaly_scores(ood_loader)

print('\n\nLSUN Detection')
show_performance(out_max_prob, in_max_prob, method_name='Baseline')


# /////////////// CIFAR data ///////////////

if args.dataset == 'cifar10':
    ood_data = dset.CIFAR100('/share/data/vision-greg/cifarpy', train=False, transform=test_transform, download=False)
    ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True,
                                             num_workers=args.prefetch, pin_memory=True)
else:
    ood_data = dset.CIFAR10('/share/data/vision-greg/cifarpy', train=False, transform=test_transform, download=False)
    ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True,
                                             num_workers=args.prefetch, pin_memory=True)

out_max_prob = get_anomaly_scores(ood_loader)

print('\n\nCIFAR-100 Detection') if num_classes == 10 else print('\n\nCIFAR-10 Detection')
show_performance(out_max_prob, in_max_prob, method_name='Baseline')

