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
from models.convnet import ConvNet
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

parser = argparse.ArgumentParser(description='Evaluates an MNIST OOD Detector',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# Setup
parser.add_argument('--test_bs', type=int, default=200)
parser.add_argument('--num_to_avg', type=int, default=1, help='Average measures across num_to_avg runs.')
parser.add_argument('--validate', '-v', action='store_true', help='Evaluate performance on validation distributions.')
parser.add_argument('--use_xent', '-x', action='store_true', help='Use cross entropy scoring instead of the MSP.')
parser.add_argument('--method_name', '-m', type=str, default='baseline', help='Method name.')
# Loading details
parser.add_argument('--load', '-l', type=str, default='./snapshots', help='Checkpoint path to resume / test.')
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
parser.add_argument('--prefetch', type=int, default=2, help='Pre-fetching threads.')
args = parser.parse_args()

# torch.manual_seed(1)
# np.random.seed(1)

test_data = dset.MNIST('/home-nfs/dan/cifar_data/mnist', train=False, transform=trn.ToTensor())
num_classes = 10

test_loader = torch.utils.data.DataLoader(
    test_data, batch_size=args.test_bs, shuffle=False,
    num_workers=args.prefetch, pin_memory=True)

# Create model
net = ConvNet()


start_epoch = 0

# Restore model
if args.load != '':
    for i in range(300 - 1, -1, -1):
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

# /////////////// Detection Prelims ///////////////

ood_num_examples = test_data.data.size(0) // 5
expected_ap = ood_num_examples / (ood_num_examples + test_data.data.size(0))

concat = lambda x: np.concatenate(x, axis=0)
to_np = lambda x: x.data.cpu().numpy()


def get_ood_scores(loader, in_dist=False):
    _score = []
    _right_score = []
    _wrong_score = []

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loader):
            if batch_idx >= ood_num_examples // args.test_bs and in_dist is False:
                break

            data = data.view(-1, 1, 28, 28).cuda()

            output = net(data)
            smax = to_np(F.softmax(output, dim=1))

            if args.use_xent:
                _score.append(to_np((output.mean(1) - torch.logsumexp(output, dim=1))))
            else:
                _score.append(-np.max(smax, axis=1))

            if in_dist:
                preds = np.argmax(smax, axis=1)
                targets = target.numpy().squeeze()
                right_indices = preds == targets
                wrong_indices = np.invert(right_indices)

                if args.use_xent:
                    _right_score.append(to_np((output.mean(1) - torch.logsumexp(output, dim=1)))[right_indices])
                    _wrong_score.append(to_np((output.mean(1) - torch.logsumexp(output, dim=1)))[wrong_indices])
                else:
                    _right_score.append(-np.max(smax[right_indices], axis=1))
                    _wrong_score.append(-np.max(smax[wrong_indices], axis=1))
    
    if in_dist:
        return concat(_score).copy(), concat(_right_score).copy(), concat(_wrong_score).copy()
    else:
        return concat(_score)[:ood_num_examples].copy()


in_score, right_score, wrong_score = get_ood_scores(test_loader, in_dist=True)

num_right = len(right_score)
num_wrong = len(wrong_score)
print('Error Rate {:.2f}'.format(100*num_wrong/(num_wrong + num_right)))

# /////////////// End Detection Prelims ///////////////

print('\nUsing MNIST as typical data')

# /////////////// Error Detection ///////////////

print('\n\nError Detection')
show_performance(wrong_score, right_score, method_name=args.method_name)

# /////////////// OOD Detection ///////////////

auroc_list, aupr_list, fpr_list = [], [], []


def get_and_print_results(ood_loader, num_to_avg=args.num_to_avg):

    aurocs, auprs, fprs = [], [], []
    for _ in range(num_to_avg):
        out_score = get_ood_scores(ood_loader)
        measures = get_measures(out_score, in_score)
        aurocs.append(measures[0]); auprs.append(measures[1]); fprs.append(measures[2])

    auroc = np.mean(aurocs); aupr = np.mean(auprs); fpr = np.mean(fprs)
    auroc_list.append(auroc); aupr_list.append(aupr); fpr_list.append(fpr)

    if num_to_avg >= 5:
        print_measures_with_std(aurocs, auprs, fprs, args.method_name)
    else:
        print_measures(auroc, aupr, fpr, args.method_name)


# /////////////// Gaussian Noise ///////////////

dummy_targets = torch.ones(ood_num_examples*args.num_to_avg)
ood_data = torch.from_numpy(
    np.clip(np.random.normal(size=(ood_num_examples*args.num_to_avg, 1, 28, 28),
                             loc=0.5, scale=0.5).astype(np.float32), 0, 1))
ood_data = torch.utils.data.TensorDataset(ood_data, dummy_targets)
ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True)

print('\n\nGaussian Noise (mu = sigma = 0.5) Detection')
get_and_print_results(ood_loader)

# /////////////// Bernoulli Noise ///////////////

dummy_targets = torch.ones(ood_num_examples*args.num_to_avg)
ood_data = torch.from_numpy(np.random.binomial(
    n=1, p=0.5, size=(ood_num_examples*args.num_to_avg, 1, 28, 28)).astype(np.float32))
ood_data = torch.utils.data.TensorDataset(ood_data, dummy_targets)
ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True)

print('\n\nBernoulli Noise Detection')
get_and_print_results(ood_loader)

# /////////////// CIFAR data ///////////////

ood_data = dset.CIFAR10(
    '/share/data/vision-greg/cifarpy', train=False,
    transform=trn.Compose([trn.Resize(28),
                           trn.Lambda(lambda x: x.convert('L', (0.2989, 0.5870, 0.1140, 0))),
                           trn.ToTensor()]))
ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True,
                                         num_workers=args.prefetch, pin_memory=True)

print('\n\nCIFAR-10 Detection')
get_and_print_results(ood_loader)

# /////////////// Icons-50 ///////////////

ood_data = dset.ImageFolder('/share/data/vision-greg/DistortedImageNet/Icons-50',
                            transform=trn.Compose([trn.Resize((28, 28)),
                                                   trn.Lambda(lambda x: x.convert('L', (0.2989, 0.5870, 0.1140, 0))),
                                                   trn.ToTensor()]))

filtered_imgs = []
for img in ood_data.imgs:
    if 'numbers' not in img[0]:     # img[0] is image name
        filtered_imgs.append(img)
ood_data.imgs = filtered_imgs

ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True)

print('\n\nIcons-50 Detection')
get_and_print_results(ood_loader)

# /////////////// Fashion-MNIST ///////////////

ood_data = dset.FashionMNIST('/share/data/vision-greg/fashion_mnist', train=False,
                             transform=trn.ToTensor(), download=False)
ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True)

print('\n\nFashion-MNIST Detection')
get_and_print_results(ood_loader)

# /////////////// Negative MNIST ///////////////

ood_data = dset.MNIST('/home-nfs/dan/cifar_data/mnist', train=False,
                      transform=trn.Compose([trn.ToTensor(), trn.Lambda(lambda img: 1 - img)]))

ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True)

print('\n\nNegative MNIST Detection')
get_and_print_results(ood_loader)

# /////////////// notMNIST ///////////////

pickle_file = '/share/data/vision-greg2/users/dan/datasets/notMNIST.pickle'
with open(pickle_file, 'rb') as f:
    notMNIST_data = pickle.load(f, encoding='latin1')
    notMNIST_data = notMNIST_data['test_dataset'].reshape((-1, 28 * 28)) + 0.5

dummy_targets = torch.ones(min(ood_num_examples*args.num_to_avg, notMNIST_data.shape[0]))
ood_data = torch.utils.data.TensorDataset(torch.from_numpy(
    notMNIST_data[:ood_num_examples*args.num_to_avg].astype(np.float32)), dummy_targets)
ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True)

print('\n\nnotMNIST Detection')
get_and_print_results(ood_loader)

# /////////////// Omniglot ///////////////

import scipy.io as sio
import scipy.misc as scimisc

# other alphabets have characters which look like digits
safe_list = [0, 2, 5, 6, 8, 12, 13, 14, 15, 16, 17, 18, 19, 21, 26]
m = sio.loadmat("/share/data/vision-greg2/users/dan/datasets/omniglot.mat")

squished_set = []
for safe_number in safe_list:
    for alphabet in m['images'][safe_number]:
        for letters in alphabet:
            for letter in letters:
                for example in letter:
                    squished_set.append(scimisc.imresize(1 - example[0], (28, 28)).reshape(1, 28 * 28))

omni_images = np.concatenate(squished_set, axis=0)

dummy_targets = torch.ones(min(ood_num_examples*args.num_to_avg, len(omni_images)))
ood_data = torch.utils.data.TensorDataset(torch.from_numpy(
    omni_images[:ood_num_examples*args.num_to_avg].astype(np.float32)), dummy_targets)
ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True)

print('\n\nOmniglot Detection')
get_and_print_results(ood_loader)

# /////////////// Mean Results ///////////////

print('\n\nMean Test Results')
print_measures(np.mean(auroc_list), np.mean(aupr_list), np.mean(fpr_list), method_name=args.method_name)


# /////////////// OOD Detection of Validation Distributions ///////////////

if args.validate is False:
    exit()

auroc_list, aupr_list, fpr_list = [], [], []

# /////////////// Uniform Noise ///////////////

dummy_targets = torch.ones(ood_num_examples*args.num_to_avg)
ood_data = torch.from_numpy(
    np.random.uniform(size=(ood_num_examples*args.num_to_avg, 1, 28, 28),
                      low=0.0, high=1.0).astype(np.float32))
ood_data = torch.utils.data.TensorDataset(ood_data, dummy_targets)
ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True)

print('\n\nUniform[0,1] Noise Detection')
get_and_print_results(ood_loader)

# /////////////// Arithmetic Mean of Images ///////////////


class AvgOfPair(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.shuffle_indices = np.arange(len(dataset))
        np.random.shuffle(self.shuffle_indices)

    def __getitem__(self, i):
        random_idx = np.random.choice(len(self.dataset))
        while random_idx == i:
            random_idx = np.random.choice(len(self.dataset))

        return self.dataset[i][0]/2. + self.dataset[random_idx][0]/2., 0

    def __len__(self):
        return len(self.dataset)


ood_loader = torch.utils.data.DataLoader(
    AvgOfPair(test_data), batch_size=args.test_bs, shuffle=True,
    num_workers=args.prefetch, pin_memory=True)

print('\n\nArithmetic Mean of Random Image Pair Detection')
get_and_print_results(ood_loader)

# /////////////// Geometric Mean of Images ///////////////


class GeomMeanOfPair(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.shuffle_indices = np.arange(len(dataset))
        np.random.shuffle(self.shuffle_indices)

    def __getitem__(self, i):
        random_idx = np.random.choice(len(self.dataset))
        while random_idx == i:
            random_idx = np.random.choice(len(self.dataset))

        return torch.sqrt(self.dataset[i][0] * self.dataset[random_idx][0]), 0

    def __len__(self):
        return len(self.dataset)


ood_loader = torch.utils.data.DataLoader(
    GeomMeanOfPair(test_data), batch_size=args.test_bs, shuffle=True,
    num_workers=args.prefetch, pin_memory=True)

print('\n\nGeometric Mean of Random Image Pair Detection')
get_and_print_results(ood_loader)

# /////////////// Jigsaw Images ///////////////

ood_loader = torch.utils.data.DataLoader(test_data, batch_size=args.test_bs, shuffle=True,
                                         num_workers=args.prefetch, pin_memory=True)

jigsaw = lambda x: torch.cat((
    torch.cat((torch.cat((x[:, 6:14, :14], x[:, :6, :14]), 1),
               x[:, 14:, :14]), 2),
    torch.cat((x[:, 14:, 14:],
               torch.cat((x[:, :14, 22:], x[:, :14, 14:22]), 2)), 2),
), 1)


ood_loader.dataset.transform = trn.Compose([trn.ToTensor(), jigsaw])

print('\n\nJigsawed Images Detection')
get_and_print_results(ood_loader)

# /////////////// Speckled Images ///////////////

speckle = lambda x: torch.clamp(x + x * torch.randn_like(x), 0, 1)
ood_loader.dataset.transform = trn.Compose([trn.ToTensor(), speckle])

print('\n\nSpeckle Noised Images Detection')
get_and_print_results(ood_loader)

# /////////////// Pixelated Images ///////////////

pixelate = lambda x: x.resize((int(28 * 0.2), int(28 * 0.2)), PILImage.BOX).resize((28, 28), PILImage.BOX)
ood_loader.dataset.transform = trn.Compose([pixelate, trn.ToTensor()])

print('\n\nPixelate Detection')
get_and_print_results(ood_loader)

# /////////////// Mirrored MNIST digits ///////////////

idxs = test_data.targets
vert_idxs = np.squeeze(np.logical_and(idxs != 3, np.logical_and(idxs != 0, np.logical_and(idxs != 1, idxs != 8))))
vert_digits = test_data.data.numpy()[vert_idxs][:, ::-1, :]

horiz_idxs = np.squeeze(np.logical_and(idxs != 0, np.logical_and(idxs != 1, idxs != 8)))
horiz_digits = test_data.data.numpy()[horiz_idxs][:, :, ::-1]

flipped_digits = concat((vert_digits, horiz_digits))

dummy_targets = torch.ones(flipped_digits.shape[0])
ood_data = torch.from_numpy(flipped_digits.astype(np.float32) / 255)
ood_data = torch.utils.data.TensorDataset(ood_data, dummy_targets)
ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True,
                                         num_workers=args.prefetch)

print('\n\nMirrored MNIST Digit Detection')
get_and_print_results(ood_loader)

# /////////////// Mean Results ///////////////

print('\n\nMean Validation Results')
print_measures(np.mean(auroc_list), np.mean(aupr_list), np.mean(fpr_list), method_name=args.method_name)
