"""
Code refactoring
* merge with conv6
* use input image size and calculate size to the first fc by knowning number of maxpool.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

from .binary_control import ControlBinaryLinear, ControlBinaryConv2d
from .networkbinarization import NetworkBinarization
from .layercontrol import LayerControl


class Vgg9Binary(nn.Module, NetworkBinarization, LayerControl):
    """Vgg-9
    VGG-inspired. 9 layers.
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
                 block1_ch=64, block2_ch=128, block3_ch=256, fc_units=256, n_classes=10,
                 epochs_per_layer=100, all_binary=False, layers_json=None):
        super(Vgg9Binary, self).__init__()
        self.features = nn.Sequential(
            ControlBinaryConv2d(3, block1_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(block1_ch),
            nn.ReLU(inplace=True),
            ControlBinaryConv2d(block1_ch, block1_ch,
                                kernel_size=3, padding=1),
            nn.BatchNorm2d(block1_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            ControlBinaryConv2d(block1_ch, block2_ch,
                                kernel_size=3, padding=1),
            nn.BatchNorm2d(block2_ch),
            nn.ReLU(inplace=True),
            ControlBinaryConv2d(block2_ch, block2_ch,
                                kernel_size=3, padding=1),
            nn.BatchNorm2d(block2_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            ControlBinaryConv2d(block2_ch, block3_ch,
                                kernel_size=3, padding=1),
            nn.BatchNorm2d(block3_ch),
            nn.ReLU(inplace=True),
            ControlBinaryConv2d(block3_ch, block3_ch,
                                kernel_size=3, padding=1),
            nn.BatchNorm2d(block3_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            ControlBinaryLinear(4*4*block3_ch, fc_units),
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
