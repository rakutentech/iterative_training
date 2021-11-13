from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

from .binary_control import ControlBinaryLinear, ControlBinaryConv2d
from .networkbinarization import NetworkBinarization
from .layercontrol import LayerControl


class Mlp3(nn.Module):
    """Fully-connected network with 3 total layers.

    Parameters
    ----------
    in_features : int
        Number of channels for input to 1st layer.
    layer_features1 : int
        Number of channels for output of 1st layer.
    layer_features2 : int
        Number of channels for output of 2nd layer.
    out_features : int
        Number of classes at the output (3rd layer).
    """

    def __init__(self,
                 in_features=784,
                 layer_features1=1024,
                 layer_features2=1024,
                 out_features=10):
        super(Mlp3, self).__init__()
        self.in_features = in_features
        self.features = nn.Sequential(
            nn.Linear(in_features, layer_features1),
            nn.BatchNorm1d(layer_features1),
            nn.ReLU(inplace=True),
            nn.Linear(layer_features1, layer_features2),
            nn.BatchNorm1d(layer_features2),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(layer_features2, out_features),
            nn.BatchNorm1d(out_features),
            nn.LogSoftmax(dim=1),
        )
        self.criterion = F.nll_loss

    def forward(self, x):
        x = x.view(-1, self.in_features)
        x = self.features(x)
        return self.classifier(x)


class Mlp3Binary(nn.Module, NetworkBinarization, LayerControl):
    """Binary version of Mlp3

    Parameters
    ----------
    in_features : int
        Number of channels for input to 1st layer.
    layer_features1 : int
        Number of channels for output of 1st layer.
    layer_features2 : int
        Number of channels for output of 2nd layer.
    out_features : int
        Number of classes at the output (3rd layer).
    epochs_per_layer : int
        Number of epochs before binarizing next layer.
    all_binary : bool
        Set true force all binary weights. Default: False.
    layers_json : str
        File to JSON that contains layers indices.
    """

    def __init__(self,
                 in_features=784,
                 layer_features1=1024,
                 layer_features2=1024,
                 out_features=10,
                 epochs_per_layer=100, all_binary=False, layers_json=None):
        super(Mlp3Binary, self).__init__()
        self.in_features = in_features
        self.features = nn.Sequential(
            ControlBinaryLinear(in_features, layer_features1),
            nn.BatchNorm1d(layer_features1),
            nn.ReLU(inplace=True),
            ControlBinaryLinear(layer_features1, layer_features2),
            nn.BatchNorm1d(layer_features2),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            ControlBinaryLinear(layer_features2, out_features),
            nn.BatchNorm1d(out_features),
            nn.LogSoftmax(dim=1),
        )
        self.criterion = F.nll_loss
        self.layer_init(epochs_per_layer=epochs_per_layer,
                        all_binary=all_binary, layers_json=layers_json)

    def forward(self, x):
        x = x.view(-1, self.in_features)
        x = self.features(x)
        return self.classifier(x)
