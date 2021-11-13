from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class NetworkBinarization:
  """A helper class for binarization for the entire network.
  """

  def restore_full_precision(self):
    for _, m in enumerate(self.modules()):
      #print(idx, '->', m)
      if hasattr(m, 'restore_full_precision_'):
        m.restore_full_precision_()

  def restore_binary_weight(self):
    for _, m in enumerate(self.modules()):
      #print(idx, '->', m)
      if hasattr(m, 'restore_binary_weight_'):
        m.restore_binary_weight_()

  def clip_weights(self):
    pass

  def report_weight_stats(self):
    maximum = 0.0
    minimum = 0.0
    num_zeros = 0
    num_binary = 0
    for _, m in enumerate(self.modules()):
      #print(idx, '->', m)
      if hasattr(m, '_report_weight_stats'):
        max_now, min_now, zeros_now, binary_now = m._report_weight_stats()
        if max_now > maximum:
          maximum = max_now
        if min_now < minimum:
          minimum = min_now
        num_zeros += zeros_now
        num_binary += binary_now
    return maximum, minimum, num_zeros, num_binary
