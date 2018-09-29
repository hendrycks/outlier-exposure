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
from models.allconv import AllConvNet
from models.wrn import WideResNet
from skimage.filters import gaussian as gblur
from PIL import Image as PILImage

# go through rigamaroo to do ...utils.display_results import show_performance
if __package__ is None:
    import sys
    from os import path

    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    from utils.display_results import show_performance, get_measures, print_measures, print_measures_with_std
    import utils.svhn_loader as svhn
    import utils.lsun_loader as lsun_loader
    from utils.validation_dataset import validation_split
    from utils.calibration_tools import *

parser = argparse.ArgumentParser(description='Evaluates a CIFAR OOD Detector',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# Setup
parser.add_argument('--test_bs', type=int, default=200)
parser.add_argument('--num_to_avg', type=int, default=1, help='Average measures across num_to_avg runs.')
parser.add_argument('--validate', '-v', action='store_true', help='Evaluate performance on validation distributions.')
parser.add_argument('--method_name', '-m', type=str, default='cifar10_allconv_calib_baseline', help='Method name.')
parser.add_argument('--use_01', '-z', action='store_true', help='Use 0-1 Posterior Rescaling.')
# Loading details
parser.add_argument('--layers', default=40, type=int, help='total number of layers')
parser.add_argument('--widen-factor', default=2, type=int, help='widen factor')
parser.add_argument('--droprate', default=0.3, type=float, help='dropout probability')
parser.add_argument('--load', '-l', type=str, default='./snapshots', help='Checkpoint path to resume / test.')
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
parser.add_argument('--prefetch', type=int, default=4, help='Pre-fetching threads.')
args = parser.parse_args()

# torch.manual_seed(1)
# np.random.seed(1)

# mean and standard deviation of channels of CIFAR-10 images
mean = [x / 255 for x in [125.3, 123.0, 113.9]]
std = [x / 255 for x in [63.0, 62.1, 66.7]]

test_transform = trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)])

if 'cifar10_' in args.method_name:
    train_data = dset.CIFAR10('/share/data/vision-greg/cifarpy', train=True, transform=test_transform)
    test_data = dset.CIFAR10('/share/data/vision-greg/cifarpy', train=False, transform=test_transform)
    num_classes = 10
else:
    train_data = dset.CIFAR100('/share/data/vision-greg/cifarpy', train=True, transform=test_transform)
    test_data = dset.CIFAR100('/share/data/vision-greg/cifarpy', train=False, transform=test_transform)
    num_classes = 100

train_data, val_data = validation_split(train_data, val_share=0.1)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.test_bs, shuffle=False,
                                         num_workers=args.prefetch, pin_memory=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.test_bs, shuffle=False,
                                          num_workers=args.prefetch, pin_memory=True)

# Create model
if 'allconv' in args.method_name:
    net = AllConvNet(num_classes)
else:
    net = WideResNet(args.layers, num_classes, args.widen_factor, dropRate=args.droprate)

start_epoch = 0

# Restore model
if args.load != '':
    for i in range(1000 - 1, -1, -1):
        if 'baseline' in args.method_name:
            subdir = 'baseline'
        elif 'oe_tune' in args.method_name:
            subdir = 'oe_tune'
        else:
            subdir = 'oe_scratch'

        model_name = os.path.join(os.path.join(args.load, subdir), args.method_name + '_epoch_' + str(i) + '.pt')
        if os.path.isfile(model_name):
            net.load_state_dict(torch.load(model_name))
            print('Model restored! Epoch:', i)
            start_epoch = i + 1
            break
    if start_epoch == 0:
        assert False, "could not resume"

net.eval()

if args.ngpu > 1:
    net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))

if args.ngpu > 0:
    net.cuda()
    # torch.cuda.manual_seed(1)

cudnn.benchmark = True  # fire on all cylinders

# /////////////// Calibration Prelims ///////////////

ood_num_examples = len(test_data) // 5
expected_ap = ood_num_examples / (ood_num_examples + len(test_data))

concat = lambda x: np.concatenate(x, axis=0)
to_np = lambda x: x.data.to('cpu').numpy()


def get_net_results(data_loader, in_dist=False, t=1):
    logits = []
    confidence = []
    correct = []
    labels = []

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            if batch_idx >= ood_num_examples // args.test_bs and in_dist is False:
                break
            data, target = data.cuda(), target.cuda()

            output = net(data)

            logits.extend(to_np(output).squeeze())

            if args.use_01:
                confidence.extend(to_np(
                    (F.softmax(output/t, dim=1).max(1)[0] - 1./num_classes)/(1 - 1./num_classes)
                ).squeeze().tolist())
            else:
                confidence.extend(to_np(F.softmax(output/t, dim=1).max(1)[0]).squeeze().tolist())

            if in_dist:
                pred = output.data.max(1)[1]
                correct.extend(pred.eq(target).to('cpu').numpy().squeeze().tolist())
                labels.extend(target.to('cpu').numpy().squeeze().tolist())

    if in_dist:
        return logits.copy(), confidence.copy(), correct.copy(), labels.copy()
    else:
        return logits[:ood_num_examples].copy(), confidence[:ood_num_examples].copy()


val_logits, val_confidence, val_correct, val_labels = get_net_results(val_loader, in_dist=True)

print('\nTuning Softmax Temperature')
t_star = tune_temp(val_logits, val_labels)
print('Softmax Temperature Tuned. Temperature is {:.3f}'.format(t_star))

test_logits, test_confidence, test_correct, _ = get_net_results(test_loader, in_dist=True, t=t_star)

print('Error Rate {:.2f}'.format(100*(len(test_correct) - sum(test_correct))/len(test_correct)))

# /////////////// End Calibration Prelims ///////////////

print('\nUsing CIFAR-10 as typical data') if num_classes == 10 else print('\nUsing CIFAR-100 as typical data')

# /////////////// In-Distribution Calibration ///////////////

print('\n\nIn-Distribution Data')
show_calibration_results(np.array(test_confidence), np.array(test_correct), method_name=args.method_name)

# /////////////// OOD Calibration ///////////////
rms_list, mad_list, sf1_list = [], [], []


def get_and_print_results(ood_loader, num_to_avg=args.num_to_avg):

    rmss, mads, sf1s = [], [], []
    for _ in range(num_to_avg):
        out_logits, out_confidence = get_net_results(ood_loader, t=t_star)

        measures = get_measures(
            concat([out_confidence, test_confidence]),
            concat([np.zeros(len(out_confidence)), test_correct]))

        rmss.append(measures[0]); mads.append(measures[1]); sf1s.append(measures[2])

    rms = np.mean(rmss); mad = np.mean(mads); sf1 = np.mean(sf1s)
    rms_list.append(rms); mad_list.append(mad); sf1_list.append(sf1)

    if num_to_avg >= 5:
        print_measures_with_std(rmss, mads, sf1s, args.method_name)
    else:
        print_measures(rms, mad, sf1, args.method_name)


# /////////////// Gaussian Noise ///////////////

dummy_targets = torch.ones(ood_num_examples * args.num_to_avg)
ood_data = torch.from_numpy(np.float32(np.clip(
    np.random.normal(size=(ood_num_examples * args.num_to_avg, 3, 32, 32), scale=0.5), -1, 1)))
ood_data = torch.utils.data.TensorDataset(ood_data, dummy_targets)
ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True,
                                         num_workers=args.prefetch, pin_memory=True)

print('\n\nGaussian Noise (sigma = 0.5) Calibration')
get_and_print_results(ood_loader)

# /////////////// Rademacher Noise ///////////////

dummy_targets = torch.ones(ood_num_examples * args.num_to_avg)
ood_data = torch.from_numpy(np.random.binomial(
    n=1, p=0.5, size=(ood_num_examples * args.num_to_avg, 3, 32, 32)).astype(np.float32)) * 2 - 1
ood_data = torch.utils.data.TensorDataset(ood_data, dummy_targets)
ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True)

print('\n\nRademacher Noise Calibration')
get_and_print_results(ood_loader)

# /////////////// Blob ///////////////

ood_data = np.float32(np.random.binomial(n=1, p=0.7, size=(ood_num_examples * args.num_to_avg, 32, 32, 3)))
for i in range(ood_num_examples * args.num_to_avg):
    ood_data[i] = gblur(ood_data[i], sigma=1.5, multichannel=False)
    ood_data[i][ood_data[i] < 0.75] = 0.0

dummy_targets = torch.ones(ood_num_examples * args.num_to_avg)
ood_data = torch.from_numpy(ood_data.transpose((0, 3, 1, 2))) * 2 - 1
ood_data = torch.utils.data.TensorDataset(ood_data, dummy_targets)
ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True,
                                         num_workers=args.prefetch, pin_memory=True)

print('\n\nBlob Calibration')
get_and_print_results(ood_loader)

# /////////////// Textures ///////////////

ood_data = dset.ImageFolder(root="/share/data/vision-greg2/users/dan/datasets/dtd/images",
                            transform=trn.Compose([trn.Resize(32), trn.CenterCrop(32),
                                                   trn.ToTensor(), trn.Normalize(mean, std)]))
ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True,
                                         num_workers=args.prefetch, pin_memory=True)

print('\n\nTexture Calibration')
get_and_print_results(ood_loader)

# /////////////// SVHN ///////////////

ood_data = svhn.SVHN(root='/share/data/vision-greg/svhn/', split="test",
                     transform=trn.Compose([trn.Resize(32), trn.ToTensor(), trn.Normalize(mean, std)]), download=False)
ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True,
                                         num_workers=args.prefetch, pin_memory=True)

print('\n\nSVHN Calibration')
get_and_print_results(ood_loader)

# /////////////// Places365 ///////////////

ood_data = dset.ImageFolder(root="/share/data/vision-greg2/places365/test_subset",
                            transform=trn.Compose([trn.Resize(32), trn.CenterCrop(32),
                                                   trn.ToTensor(), trn.Normalize(mean, std)]))
ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True,
                                         num_workers=args.prefetch, pin_memory=True)

print('\n\nPlaces365 Calibration')
get_and_print_results(ood_loader)

# /////////////// LSUN ///////////////

ood_data = lsun_loader.LSUN("/share/data/vision-greg2/users/dan/datasets/LSUN/lsun-master/data", classes='test',
                            transform=trn.Compose([trn.Resize(32), trn.CenterCrop(32),
                                                   trn.ToTensor(), trn.Normalize(mean, std)]))
ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True,
                                         num_workers=args.prefetch, pin_memory=True)

print('\n\nLSUN Calibration')
get_and_print_results(ood_loader)

# /////////////// CIFAR Data ///////////////

if 'cifar10_' in args.method_name:
    ood_data = dset.CIFAR100('/share/data/vision-greg/cifarpy', train=False, transform=test_transform)
else:
    ood_data = dset.CIFAR10('/share/data/vision-greg/cifarpy', train=False, transform=test_transform)

ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True,
                                         num_workers=args.prefetch, pin_memory=True)


print('\n\nCIFAR-100 Detection') if 'cifar100' in args.method_name else print('\n\nCIFAR-10 Detection')
get_and_print_results(ood_loader)

# /////////////// Mean Results ///////////////

print('\n\nMean Test Results')
print_measures(np.mean(rms_list), np.mean(mad_list), np.mean(sf1_list), method_name=args.method_name)
