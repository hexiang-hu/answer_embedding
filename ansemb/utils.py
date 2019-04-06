import os
import json

import torch
import torch.nn as nn
import torchvision.transforms as transforms

from ansemb.config import cfg

def cosine_sim(im, s):
  return im.mm(s.t())

def batch_mc_acc(predicted):
  """ Compute the accuracies for a batch of predictions and answers """
  N, C = predicted.squeeze().size()
  _, predicted_index = predicted.max(dim=1, keepdim=True)
  return (predicted_index == C - 1).float()

def batch_top1(predicted, true):
  """ Compute the accuracies for a batch of predictions and answers """
  _, predicted_index = predicted.max(dim=1, keepdim=True)
  return true.gather(dim=1, index=predicted_index).clamp(max=1)

def batch_accuracy(predicted, true):
  """ Compute the accuracies for a batch of predictions and answers """
  _, predicted_index = predicted.max(dim=1, keepdim=True)
  agreeing = true.gather(dim=1, index=predicted_index)
  return (agreeing * 0.3).clamp(max=1)

def update_learning_rate(optimizer, epoch):
  learning_rate = cfg.TRAIN.base_lr * 0.5**(float(epoch) / cfg.TRAIN.lr_decay)
  for param_group in optimizer.param_groups: param_group['lr'] = learning_rate

  return learning_rate

class Tracker:
  """ Keep track of results over time, while having access to monitors to display information about them. """
  def __init__(self):
    self.data = {}

  def track(self, name, *monitors):
    """ Track a set of results with given monitors under some name (e.g. 'val_acc').
        When appending to the returned list storage, use the monitors to retrieve useful information.
    """
    l = Tracker.ListStorage(monitors)
    self.data.setdefault(name, []).append(l)
    return l

  def to_dict(self):
    # turn list storages into regular lists
    return {k: list(map(list, v)) for k, v in self.data.items()}

  class ListStorage:
    """ Storage of data points that updates the given monitors """
    def __init__(self, monitors=[]):
      self.data = []
      self.monitors = monitors
      for monitor in self.monitors:
        setattr(self, monitor.name, monitor)

    def append(self, item):
      for monitor in self.monitors:
        monitor.update(item)
      self.data.append(item)

    def __iter__(self):
        return iter(self.data)

  class MeanMonitor:
    """ Take the mean over the given values """
    name = 'mean'

    def __init__(self):
      self.n = 0
      self.total = 0

    def update(self, value):
      self.total += value
      self.n += 1

    @property
    def value(self):
      return self.total / self.n

  class MovingMeanMonitor:
    """ Take an exponentially moving mean over the given values """
    name = 'mean'

    def __init__(self, momentum=0.9):
      self.momentum = momentum
      self.first = True
      self.value = None

    def update(self, value):
      if self.first:
        self.value = value
        self.first = False
      else:
        m = self.momentum
        self.value = m * self.value + (1 - m) * value

def get_transform(target_size, central_fraction=1.0):
  return transforms.Compose([
      transforms.Scale(int(target_size / central_fraction)),
      transforms.CenterCrop(target_size),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
  ])
