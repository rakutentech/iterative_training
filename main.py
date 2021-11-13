from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import random
import sys
import time

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter

"""
try:
    from tensorboardX import SummaryWriter
except ImportError:
    raise ImportError('tensorboardX must be installed. Please see README.')
"""

from train import train
import util


# Training settings
parser = argparse.ArgumentParser(description='PyTorch Training')

parser.add_argument('--binary-p', type=float, default=0.1, metavar='BP',
                    help='Percentage to binarize (default: 0.1)')

parser.add_argument('--batch-size', type=int, default=100, metavar='BS',
                    help='input batch size for training (default: 100)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='TBS',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='E',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--patience', type=int, default=0, metavar='P',
                    help='Patient for new accuracies (default: 0 off)')
parser.add_argument('--lr', type=float, default=0.01, metavar='OLR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='OM',
                    help='SGD momentum (default: 0.0)')
parser.add_argument('--scheduler-milestones', default='[150,250]', metavar='SM',
                    help='Epoch schedule to reduce learning rate')
parser.add_argument('--scheduler-gamma', type=float, default=0.1, metavar='SG',
                    help='Learning rate reduction scaling (default: 0.1)')
"""
parser.add_argument('--lr-gamma', type=float, default=1.0, metavar='OLRG',
                    help='learning rate gamma (default: 1.0 ie off)(set to less than 1.0)')
"""
"""
BinaryConnect LR_decay
https://github.com/MatthieuCourbariaux/BinaryConnect/blob/lasagne/mnist.py
num_epochs=250
LR_start = .001
LR_fin = 0.000003
LR_decay = (LR_fin/LR_start)**(1./num_epochs)
"""

parser.add_argument('--seed', type=int, default=316, metavar='S',
                    help='random seed (default: 316)')
parser.add_argument('--log-interval', type=int, default=20, metavar='LI',
                    help='how many batches to wait before logging training status')
parser.add_argument('--num-processes', type=int, default=2, metavar='N',
                    help='how many training processes to use (default: 2)')
parser.add_argument('--gpu', type=int, default=0, metavar='G',
                    help='which GPU to use (default: 0)')
parser.add_argument('--logdir', default='./logs', metavar='L',
                    help='directory to write logs (default: ./logs)')

parser.add_argument('--model', default='mlp', metavar='M',
                    help='Choose NN model. Check load_model()')
parser.add_argument('--dataset', default='mnist', metavar='D',
                    help='Choose dataset: mnist (default), cifar10, svhn')
parser.add_argument('--dataset-folder', default='./data', metavar='DF',
                    help='Folder to datasets')
parser.add_argument('--use-data-augmentation', action='store_true',
                    help='Turn on data augmentation (default: off)')
parser.add_argument('--dataset-indices', default='cifar10_split0613.json',
                    help='Index file for dataset. Default: cifar10_split0613.json')

parser.add_argument('--epochs-per-layer', type=int, default=150, metavar='EPL',
                    help='Epochs per layer for layer-by-layer binarization (default: 150)')
parser.add_argument('--layers-indices', default='layers.json',
                    help='Index file for layers. Default: layers.json')
parser.add_argument('--reverse-layer-binarization', action='store_true',
                    help='Reverse layer binarization (default: off)')
parser.add_argument('--test-validation', action='store_true',
                    help='Test validation data (default: off)')
parser.add_argument('--test-training', action='store_true',
                    help='Test training dataset at each epoch (default: off)')
parser.add_argument('--record-histogram', action='store_true',
                    help='Save histogram to tensorboard (default: off)')

parser.add_argument('--save-progress', action='store_true',
                    help='Turn on save progress')
parser.add_argument('--save-last', action='store_true',
                    help='Save model at the end of training')
parser.add_argument('--load-last', action='store_true',
                    help='Save model at the end of training')
parser.add_argument('--load-last-file', default='last.tar',
                    help='Save file to load (default: last.tar)')
parser.add_argument('--verbose', action='store_true',
                    help='Turn on verbose mode')


def load_model(args):
    """Load NN model
    """

    if args.dataset == 'mnist':
        if args.model == '300':
            from models.mlp3 import Mlp3
            return Mlp3(in_features=784, layer_features1=300, layer_features2=100, out_features=10)
        elif args.model == '300_layer':
            from models.mlp3 import Mlp3Binary
            return Mlp3Binary(in_features=784, layer_features1=300, layer_features2=100, out_features=10,
                              epochs_per_layer=args.epochs_per_layer, all_binary=False, layers_json=args.layers_indices)
        elif args.model == '300_binary':
            from models.mlp3 import Mlp3Binary
            return Mlp3Binary(in_features=784, layer_features1=300, layer_features2=100, out_features=10,
                              epochs_per_layer=args.epochs_per_layer, all_binary=True, layers_json=args.layers_indices)
        elif args.model == '784':
            from models.mlp3 import Mlp3
            return Mlp3(in_features=784, layer_features1=784, layer_features2=784, out_features=10)
        elif args.model == '784_layer':
            from models.mlp3 import Mlp3Binary
            return Mlp3Binary(in_features=784, layer_features1=784, layer_features2=784, out_features=10,
                              epochs_per_layer=args.epochs_per_layer, all_binary=False, layers_json=args.layers_indices)
        elif args.model == '784_binary':
            from models.mlp3 import Mlp3Binary
            return Mlp3Binary(in_features=784, layer_features1=784, layer_features2=784, out_features=10,
                              epochs_per_layer=args.epochs_per_layer, all_binary=True, layers_json=args.layers_indices)
        else:
            raise NotImplementedError(
                'Unknown model requested: {}'.format(args.model))
    elif args.dataset == 'cifar10':
        if args.model == 'vgg5':
            from models.vgg5 import Vgg5
            return Vgg5()
        elif args.model == 'vgg5_layer':
            from models.vgg5binary import Vgg5Binary
            return Vgg5Binary(epochs_per_layer=args.epochs_per_layer, all_binary=False, layers_json=args.layers_indices)
        elif args.model == 'vgg5_binary':
            from models.vgg5binary import Vgg5Binary
            return Vgg5Binary(epochs_per_layer=args.epochs_per_layer, all_binary=True, layers_json=args.layers_indices)
        elif args.model == 'vgg9':
            from models.vgg9 import Vgg9
            return Vgg9()
        elif args.model == 'vgg9_layer':
            from models.vgg9binary import Vgg9Binary
            return Vgg9Binary(epochs_per_layer=args.epochs_per_layer, all_binary=False, layers_json=args.layers_indices)
        elif args.model == 'vgg9_binary':
            from models.vgg9binary import Vgg9Binary
            return Vgg9Binary(epochs_per_layer=args.epochs_per_layer, all_binary=True, layers_json=args.layers_indices)
        elif args.model == 'resnet20':
            from models.resnet import resnet20
            net = resnet20()
            net.criterion = nn.CrossEntropyLoss()
            return net
        elif args.model == 'resnet20_layer':
            from models.resnet_control import resnet20
            net = resnet20(epochs_per_layer=args.epochs_per_layer,
                           all_binary=False, layers_json=args.layers_indices)
            net.criterion = nn.CrossEntropyLoss()
            return net
        elif args.model == 'resnet20_binary':
            from models.resnet_control import resnet20
            net = resnet20(epochs_per_layer=args.epochs_per_layer,
                           all_binary=True, layers_json=args.layers_indices)
            net.criterion = nn.CrossEntropyLoss()
            return net
        else:
            raise NotImplementedError(
                'Unknown model requested: {}'.format(args.model))
    else:
        raise NotImplementedError(
            'Unknown dataset requested: {}'.format(args.dataset))


def add_optimizer_scheduler(args, net):
    if args.model.startswith('resnet20'):
        net.optimizer = optim.SGD(
            net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=1e-4)
        net.scheduler = optim.lr_scheduler.MultiStepLR(net.optimizer, milestones=eval(
            args.scheduler_milestones), gamma=args.scheduler_gamma)

    return net


def load_dataset(args):
    if args.dataset == 'mnist':
        data_loaders = util.dataset.load_mnist_with_validation(
            args, data_folder=args.dataset_folder)
    elif args.dataset == 'cifar10':
        data_loaders = util.dataset.load_cifar10_with_preselected_validation_fix_validation_transform(
            batch_size=args.batch_size,
            test_batch_size=args.test_batch_size,
            use_data_augmentation=args.use_data_augmentation,
            index_file=args.dataset_indices,
            data_folder=args.dataset_folder
        )
    else:
        raise NotImplementedError('Unknown dataset requested')
    return data_loaders


def save_args(args, writer):
    print("argv: {}".format(sys.argv))
    writer.add_text('argv', str(sys.argv), 0)

    print("args:")
    options = vars(args)
    for k, v in options.items():
        #print("  {} {} {} {}".format(k, v, type(k), type(v)))
        print("  {} {}".format(k, v))
        writer.add_text("args/"+k, str(v), 0)


def set_all_seed(seed):
    print("Setting all seeds to {}".format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        print("CuDNN enabled")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def show_weights(net):
  total = 0
  for idx, m in enumerate(net.modules()):
    #print(idx, '->', m)
    """
    if hasattr(m, 'weight'):
      if not isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        print(idx, '->', m, 'n_weight', m.weight.nelement())
        total += m.weight.nelement()
    """
    if isinstance(m, (nn.Linear, nn.Conv2d)):
      print(idx, '->', m, 'n_weight', m.weight.nelement())
      total += m.weight.nelement()
  print("Total number of weights: {}".format(total))


def main():
    main_start = time.time()
    args = parser.parse_args()
    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)

    set_all_seed(args.seed)

    writer = SummaryWriter(args.logdir+'/tb')
    save_args(args, writer)

    device = torch.device("cuda:"+str(args.gpu)
                          if torch.cuda.is_available() else "cpu")
    # Assume that we are on a CUDA machine, then this should print a CUDA device:
    print('Hardware: {}'.format(device))  # cuda:X
    writer.add_text('hardware', str(device), 0)

    model = load_model(args)
    model = add_optimizer_scheduler(args, model)
    print(model)
    show_weights(model)

    data_loaders = load_dataset(args)
    if torch.cuda.is_available():
        train(args, model, data_loaders, device, writer)
    else:
        print("Please run with a GPU")

    print("Closing writer...")
    writer.close()  # close to flush cache
    print('Script duration: {:.4f} hours'.format(
        (time.time() - main_start)/3600.0))




if __name__ == '__main__':
    main()
