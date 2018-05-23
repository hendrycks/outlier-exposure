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
    from utils.calibration_tools import *

parser = argparse.ArgumentParser(description='Evaluates Anomaly Detectors with a WideResNet on SVHN',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--nooe', action='store_true', help='Evaluate model without OE?')
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

train_data = svhn.SVHN(root='/share/data/vision-greg/svhn/', split="train_and_extra",
                       transform=trn.ToTensor(), download=False)
test_data = svhn.SVHN(root='/share/data/vision-greg/svhn/', split="test", transform=trn.ToTensor(), download=False)
num_classes = 10
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
val_size = val_data.length

val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.test_bs, shuffle=False,
                                         num_workers=args.prefetch, pin_memory=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.test_bs, shuffle=False,
                                          num_workers=args.prefetch, pin_memory=True)

# Create model
net = WideResNet(args.layers, num_classes, args.widen_factor, dropRate=args.droprate)

# Restore model
start_epoch = 0
if args.load != '':
    for i in range(300 - 1, -1, -1):
        if args.nooe:
            model_name = os.path.join(args.load, 'svhn_model_epoch' + str(i) + '.pytorch')
            method = 'Baseline'
        else:
            model_name = os.path.join(args.load, 'svhn_model_tuned_epoch' + str(i) + '.pytorch')
            method = '+OE'
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


# /////////////// Calibration Prelims ///////////////

ood_num_examples = test_data.test_data.shape[0] // 5
expected_ap = ood_num_examples / (ood_num_examples + test_data.test_data.shape[0])

concat = lambda x: np.concatenate(x, axis=0)
to_np = lambda x: x.data.cpu().numpy()


def get_net_results(data_loader, in_dist=False, t=1):
    logits = []
    confidence = []
    correct = []

    for batch_idx, (data, target) in enumerate(data_loader):
        if batch_idx >= ood_num_examples // args.test_bs and in_dist is False:
            break
        data, target = V(data.cuda(), volatile=True), target.long().squeeze().cuda()

        output = net(data)

        logits.extend(to_np(output).squeeze())
        # confidence.extend(to_np(F.softmax(output/t, dim=1).max(1)[0]).squeeze().tolist())
        confidence.extend(to_np(
            (F.softmax(output/t, dim=1).max(1)[0] - 1./num_classes)/(1 - 1./num_classes)
        ).squeeze().tolist())

        if in_dist:
            pred = output.data.max(1)[1]
            correct.extend(pred.eq(target).cpu().numpy().squeeze().tolist())

    if in_dist:
        return logits.copy(), confidence.copy(), correct.copy()
    else:
        return logits[:ood_num_examples].copy(), confidence[:ood_num_examples].copy()


val_logits, val_confidence, val_correct = get_net_results(val_loader, in_dist=True)

val_labels = val_data.parent_ds.train_labels[val_data.offset:]
t_star = tune_temp(val_logits, val_labels, val_correct)

test_logits, test_confidence, test_correct = get_net_results(test_loader, in_dist=True, t=t_star)

# /////////////// End Calibration Prelims ///////////////

print('Using SVHN as typical data')

# /////////////// In-Distribution Data ///////////////

print('\n\nIn-Distribution Data')
show_calibration_results(np.array(test_confidence), np.array(test_correct), method_name=method)

# /////////////// Gaussian Noise ///////////////

dummy_targets = torch.ones(ood_num_examples)
ood_data = torch.from_numpy(
    np.clip(np.random.normal(size=(ood_num_examples, 3, 32, 32), loc=0.5, scale=0.5).astype(np.float32), 0, 1))
ood_data = torch.utils.data.TensorDataset(ood_data, dummy_targets)
ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.batch_size, shuffle=True)

_, out_confidence = get_net_results(ood_loader, t=t_star)

print('\n\nGaussian Noise (mu, sigma = 0.5) Detection')
show_calibration_results(concat([out_confidence, test_confidence]),
                         concat([np.zeros(len(out_confidence)), test_correct]),
                         method_name=method)

# /////////////// Bernoulli Noise ///////////////

dummy_targets = torch.ones(ood_num_examples)
ood_data = torch.from_numpy(np.random.binomial(n=1, p=0.5, size=(ood_num_examples, 3, 32, 32)).astype(np.float32))
ood_data = torch.utils.data.TensorDataset(ood_data, dummy_targets)
ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.batch_size, shuffle=True)

_, out_confidence = get_net_results(ood_loader, t=t_star)

print('\n\nBernoulli Noise Detection')
show_calibration_results(concat([out_confidence, test_confidence]),
                         concat([np.zeros(len(out_confidence)), test_correct]),
                         method_name=method)

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

_, out_confidence = get_net_results(ood_loader, t=t_star)

print('\n\nBlob Detection')
show_calibration_results(concat([out_confidence, test_confidence]),
                         concat([np.zeros(len(out_confidence)), test_correct]),
                         method_name=method)

# /////////////// Icons-50 data ///////////////

ood_data = dset.ImageFolder('/share/data/vision-greg/DistortedImageNet/Icons-50',
                            transform=trn.Compose([trn.Resize((32, 32)),
                                                   trn.ToTensor()]))

filtered_imgs = []
for img in ood_data.imgs:
    if 'numbers' not in img[0]:     # img[0] is image name
        filtered_imgs.append(img)
ood_data.imgs = filtered_imgs

ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True,
                                         num_workers=args.prefetch, pin_memory=True)

_, out_confidence = get_net_results(ood_loader, t=t_star)

print('\n\nIcons-50 Detection')
show_calibration_results(concat([out_confidence, test_confidence]),
                         concat([np.zeros(len(out_confidence)), test_correct]),
                         method_name=method)

# /////////////// Textures ///////////////

ood_data = dset.ImageFolder(root="/share/data/lang/users/dan/datasets/dtd/images",
                            transform=trn.Compose([trn.Resize(32), trn.CenterCrop(32),
                                                   trn.ToTensor()]))
ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.batch_size, shuffle=True,
                                         num_workers=args.prefetch, pin_memory=True)

_, out_confidence = get_net_results(ood_loader, t=t_star)

print('\n\nTexture Detection')
show_calibration_results(concat([out_confidence, test_confidence]),
                         concat([np.zeros(len(out_confidence)), test_correct]),
                         method_name=method)

# /////////////// Places365 ///////////////

ood_data = dset.ImageFolder(root="/share/data/lang/users/dan/datasets/places365/test_subset",
                            transform=trn.Compose([trn.Resize(32), trn.CenterCrop(32),
                                                   trn.ToTensor()]))
ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.batch_size, shuffle=True,
                                         num_workers=args.prefetch, pin_memory=True)

_, out_confidence = get_net_results(ood_loader, t=t_star)

print('\n\nPlaces365 Detection')
show_calibration_results(concat([out_confidence, test_confidence]),
                         concat([np.zeros(len(out_confidence)), test_correct]),
                         method_name=method)

# /////////////// LSUN ///////////////

ood_data = lsun_loader.LSUN("/share/data/lang/users/dan/datasets/LSUN/lsun-master/data", classes='test',
                            transform=trn.Compose([trn.Resize(32), trn.CenterCrop(32),
                                                   trn.ToTensor()]))

ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.batch_size, shuffle=True,
                                         num_workers=args.prefetch, pin_memory=True)

_, out_confidence = get_net_results(ood_loader, t=t_star)

print('\n\nLSUN Detection')
show_calibration_results(concat([out_confidence, test_confidence]),
                         concat([np.zeros(len(out_confidence)), test_correct]),
                         method_name=method)

# /////////////// CIFAR data ///////////////

ood_data = dset.CIFAR10('~/cifar_data', train=False, transform=trn.ToTensor(), download=False)
ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True,
                                         num_workers=args.prefetch, pin_memory=True)

_, out_confidence = get_net_results(ood_loader, t=t_star)

print('\n\nCIFAR-10 Detection')
show_calibration_results(concat([out_confidence, test_confidence]),
                         concat([np.zeros(len(out_confidence)), test_correct]),
                         method_name=method)

# /////////////// Street View Characters data ///////////////

ood_data = dset.ImageFolder(root="/share/data/lang/users/dan/datasets/StreetLetters",
                            transform=trn.Compose([trn.Resize(32), trn.CenterCrop(32),
                                                   trn.ToTensor()]))
ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.batch_size, shuffle=True,
                                         num_workers=args.prefetch, pin_memory=True)

_, out_confidence = get_net_results(ood_loader, t=t_star)

print('\n\nStreet View Characters Detection')
show_calibration_results(concat([out_confidence, test_confidence]),
                         concat([np.zeros(len(out_confidence)), test_correct]),
                         method_name=method)

