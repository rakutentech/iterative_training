from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


class Vgg5(nn.Module):
    """Vgg-5
    VGG-inspired, 5 layers.
    For CIFAR10. 
    See Frankle et al, 2019 paper.

    Parameters
    ----------
    block1_ch : int
        Number of channel for VGG block 1.
    fc_units : int
        Number of units in the FC layer.
    n_classes : int
        Number of classes at the output.
    """

    def __init__(self, block1_ch=64, fc_units=256, n_classes=10):
        super(Vgg5, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, block1_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(block1_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(block1_ch, block1_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(block1_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(16384, fc_units),
            nn.BatchNorm1d(fc_units),
            nn.ReLU(inplace=True),
            nn.Linear(fc_units, fc_units),
            nn.BatchNorm1d(fc_units),
            nn.ReLU(inplace=True),
            nn.Linear(fc_units, n_classes),
            nn.BatchNorm1d(n_classes),
            nn.LogSoftmax(dim=1),
        )
        self.criterion = F.nll_loss

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out
