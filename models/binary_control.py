from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F


class BinaryController(nn.Module):
  """Randomizer according to Courbaraux et al, BinaryConnect.
  """

  def binarize_weight_(self):
    """Binarize weights
    Weights are modified in-place.
    """
    with torch.no_grad():
      self.weight_orig = self.weight.clone()
      if self.binarize_:
        self.weight.sign_()
      self.weight_binary = self.weight.clone()

  def restore_full_precision_(self):
    """Restore full-precision parameters (weights and bias) for parameter update.
    """
    with torch.no_grad():
      self.weight.copy_(self.weight_orig)

  def restore_binary_weight_(self):
    with torch.no_grad():
      self.weight.copy_(self.weight_binary)

  def _report_weight_stats(self):
    with torch.no_grad():
      maximum = self.weight.max()
      minimum = self.weight.min()
      num_zeros = torch.sum(torch.eq(self.weight, 0.0))
      if self.binarize_:
        num_binary = self.weight.nelement()
      else:
        num_binary = 0
      #print("max {} min {} zeros {}".format(maximum, minimum, num_zeros))
      return maximum, minimum, num_zeros, num_binary

  def clip_weights_(self, low=-1.0, high=1.0):
    pass


class ControlBinaryLinear(nn.Linear, BinaryController):

#  def __init__(self, in_features, out_features, bias=True):
#    super(ControlBinaryLinear, self).__init__(
  def __init__(self, *kargs, **kwargs):
    super(ControlBinaryLinear, self).__init__(*kargs, **kwargs)
    self.binarize_ = False

  def forward(self, input):
    self.binarize_weight_()
    out = F.linear(input, self.weight, self.bias)
    return out


class ControlBinaryConv2d(nn.Conv2d, BinaryController):

  def __init__(self, *kargs, **kwargs):
    super(ControlBinaryConv2d, self).__init__(*kargs, **kwargs)
    self.binarize_ = False

  def forward(self, input):
    self.binarize_weight_()
    out = F.conv2d(
      input, self.weight, self.bias, self.stride,
      self.padding, self.dilation, self.groups)
    return out


def binarizable_layers(model: nn.Module) -> list:
  """Return indices of binarizable layers
  """
  ret = []
  for idx, m in enumerate(model.modules()):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
      ret.append(idx)
      print(idx, m)
  return ret
