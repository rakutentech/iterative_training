from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

from .binary_control import ControlBinaryLinear, ControlBinaryConv2d
from .networkbinarization import NetworkBinarization
from .layercontrol import LayerControl


class Vgg5Binary(nn.Module, NetworkBinarization, LayerControl):
    """Vgg-5
    VGG-inspired, 5 layers.
    For CIFAR10. 
    See Frankle et al, 2019 paper.
    Variant: supports layer-by-layer binarization.

    Parameters
    ----------
    block1_ch : int
        Number of channel for VGG block 1.
    fc_units : int
        Number of units in the FC layer.
    n_classes : int
        Number of classes at the output.
    epochs_per_layer : int
        Number of epochs before binarizing next layer.
    all_binary : bool
        Set true force all binary weights. Default: False.
    layers_json : str
        File to JSON that contains layers indices.
    """

    def __init__(self, 
                 block1_ch=64, fc_units=256, n_classes=10,
                 epochs_per_layer=100, all_binary=False, layers_json=None):
        super(Vgg5Binary, self).__init__()
        self.features = nn.Sequential(
            ControlBinaryConv2d(3, block1_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(block1_ch),
            nn.ReLU(inplace=True),
            ControlBinaryConv2d(block1_ch, block1_ch,
                                kernel_size=3, padding=1),
            nn.BatchNorm2d(block1_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            ControlBinaryLinear(16384, fc_units),
            nn.BatchNorm1d(fc_units),
            nn.ReLU(inplace=True),
            ControlBinaryLinear(fc_units, fc_units),
            nn.BatchNorm1d(fc_units),
            nn.ReLU(inplace=True),
            ControlBinaryLinear(fc_units, n_classes),
            nn.BatchNorm1d(n_classes),
            nn.LogSoftmax(dim=1),
        )
        self.criterion = F.nll_loss
        self.layer_init(epochs_per_layer=epochs_per_layer, all_binary=all_binary, layers_json=layers_json)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out
