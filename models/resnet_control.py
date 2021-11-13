'''
Mainly based on below:
Based on https://github.com/akamaster/pytorch_resnet_cifar10
License: BSD 2-clause

Changes:
Parameter initialization is changed to PyTorch default (nn.Linear):

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

from:
  
    from init.kaiming_normal(m.weight)

In the above code, the last FC layer uses biases, so this change will
have an effect. However, not sure which variant the ResNet paper uses.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

import json

from .binary_control import ControlBinaryLinear, ControlBinaryConv2d
from .layerbinaryhelper import LayerBinaryHelper


__all__ = ['ResNet', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']

def _weights_init(m):
    #classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, ControlBinaryLinear) or isinstance(m, ControlBinaryConv2d):
        #init.kaiming_normal(m.weight)
        m.reset_parameters()
    elif isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        raise ValueError("Need to convert to Control Binary module")


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = ControlBinaryConv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = ControlBinaryConv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     ControlBinaryConv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module, LayerBinaryHelper):
    def __init__(self, block, num_blocks, num_classes=10, epochs_per_layer=100, all_binary=False, layers_json=None):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = ControlBinaryConv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = ControlBinaryLinear(64, num_classes)

        self.apply(_weights_init)

        self.layer_init(epochs_per_layer=epochs_per_layer, all_binary=all_binary, layers_json=layers_json)


    def layer_init(self, epochs_per_layer, all_binary, layers_json):
        self._all_binary = all_binary
        # Track training progress
        self.epochs_per_layer_ = epochs_per_layer
        self.epochs_ = 0
        self.nets_ = list(self.modules())
        self.layers_ = self.load_layers(layers_json)
        if self.layers_ is None:
            raise ValueError("Problem with reading layers json file: {}".format(layers_json))
        self.sanity_check()
        if all_binary:
            self.binarize_all()

    def sanity_check(self):
        # A layer that can be binarized have this variable
        for n in self.layers_:
            assert hasattr(self.nets_[n], "binarize_")

    def load_layers(self, json_file):
        with open(json_file, 'r') as fp:
            print("Loading layers from {}...".format(json_file))
            return json.load(fp)
        return None

    def detect_layers(self):
        layers = []
        test = (ControlBinaryLinear, ControlBinaryConv2d)
        for idx, m in enumerate(self.modules()):
            if isinstance(m, test):
                layers.append(idx)
                #print(idx, '->', m)
        print(len(layers), "layers", layers)
        return layers

    def show_binarized(self):
        status = ""
        layers = ""
        for n in self.layers_:
            status += "{:3}".format(self.nets_[n].binarize_)
            layers += "{:3}".format(n)
        print("Layer ", layers)
        print("Binary", status)

    def binarize_layer(self):
        """Set layer binarization on a forward-first schedule.
        Call before training an epoch.
        """
        layer = self.epochs_ // self.epochs_per_layer_
        if layer < len(self.layers_):
            self.nets_[self.layers_[layer]].binarize_ = True
        else:
            print("All already binarized")
        self.epochs_ += 1
        self.show_binarized()

    def binarize_layer_reverse(self):
        """Set layer binarization on a last-first schedule.
        Call before training an epoch.
        """
        layer = self.epochs_ // self.epochs_per_layer_
        if layer < len(self.layers_):
            self.nets_[self.layers_[-1-layer]].binarize_ = True
        else:
            print("All already binarized")
        self.epochs_ += 1
        self.show_binarized()

    def binarize_all(self):
        """Binarize all layers
        Call once before starting any training.
        """
        for n in self.layers_:
            self.nets_[n].binarize_ = True
        self.show_binarized()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def resnet20(epochs_per_layer, all_binary, layers_json):
    return ResNet(BasicBlock, [3, 3, 3], epochs_per_layer=epochs_per_layer, all_binary=all_binary, layers_json=layers_json)


def resnet32(epochs_per_layer, all_binary, layers_json):
    return ResNet(BasicBlock, [5, 5, 5], epochs_per_layer=epochs_per_layer, all_binary=all_binary, layers_json=layers_json)


def resnet44(epochs_per_layer, all_binary, layers_json):
    return ResNet(BasicBlock, [7, 7, 7], epochs_per_layer=epochs_per_layer, all_binary=all_binary, layers_json=layers_json)


def resnet56(epochs_per_layer, all_binary, layers_json):
    return ResNet(BasicBlock, [9, 9, 9], epochs_per_layer=epochs_per_layer, all_binary=all_binary, layers_json=layers_json)


def resnet110(epochs_per_layer, all_binary, layers_json):
    return ResNet(BasicBlock, [18, 18, 18], epochs_per_layer=epochs_per_layer, all_binary=all_binary, layers_json=layers_json)


def resnet1202(epochs_per_layer, all_binary, layers_json):
    return ResNet(BasicBlock, [200, 200, 200], epochs_per_layer=epochs_per_layer, all_binary=all_binary, layers_json=layers_json)


def test(net):
    import numpy as np
    total_params = 0

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params", total_params)
    print("Total layers", len(list(filter(lambda p: p.requires_grad and len(p.data.size())>1, net.parameters()))))


if __name__ == "__main__":
    for net_name in __all__:
        if net_name.startswith('resnet'):
            print(net_name)
            test(globals()[net_name]())
            print()
