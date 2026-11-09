"""Flat-input HORA RunningMeanStd semantics, adapted from the MIT code release.

Original: Haozhi Qi, 2022, hora/algo/models/running_mean_std.py.
"""

import torch
from torch import nn


class HoraRunningMeanStd(nn.Module):
  running_mean: torch.Tensor
  running_var: torch.Tensor
  count: torch.Tensor

  def __init__(self, width):
    super().__init__()
    self.register_buffer("running_mean", torch.zeros(width, dtype=torch.float64))
    self.register_buffer("running_var", torch.ones(width, dtype=torch.float64))
    self.register_buffer("count", torch.ones((), dtype=torch.float64))
    self.epsilon = 1e-5

  @torch.no_grad()
  def update(self, value):
    mean = value.mean(0)
    variance = value.var(0)
    batch_count = value.shape[0]
    delta = mean - self.running_mean
    total = self.count + batch_count
    second = self.running_var * self.count + variance * batch_count
    second += delta.square() * self.count * batch_count / total
    self.running_mean.copy_(self.running_mean + delta * batch_count / total)
    self.running_var.copy_(second / total)
    self.count.copy_(total)

  def forward(self, value, unnorm=False):
    if self.training:
      self.update(value)
    mean, std = (
      self.running_mean.float(),
      (self.running_var.float() + self.epsilon).sqrt(),
    )
    if unnorm:
      return value.clamp(-5, 5) * std + mean
    return ((value - mean) / std).clamp(-5, 5)
