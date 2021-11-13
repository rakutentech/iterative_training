# Saving & Loading a General Checkpoint for Inference and/or Resuming Training
# https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-a-general-checkpoint-for-inference-and-or-resuming-training

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import torch


def save(epoch, model, optimizer, loss, folder, pattern='{}.tar', verbose=False):
  file_name = os.path.join(folder, pattern.format(epoch))

  torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
  }, file_name)
  if verbose:
    print('Model saved to {}'.format(file_name))


def load(file_name, model, optimizer):
  checkpoint = torch.load(file_name)
  model.load_state_dict(checkpoint['model_state_dict'])
  optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
  epoch = checkpoint['epoch']
  loss = checkpoint['loss']

  #Outside
  #model.eval()
  # - or -
  #model.train()
  return epoch, loss
