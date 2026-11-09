"""HORA privileged encoder added to RSL-RL's MLP model."""

import torch
from rsl_rl.models import MLPModel
from torch import nn

from .normalization import HoraRunningMeanStd


class HoraModel(MLPModel):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.env_encoder = nn.Sequential(
      nn.Linear(9, 256),
      nn.ELU(),
      nn.Linear(256, 128),
      nn.ELU(),
      nn.Linear(128, 8),
      nn.ELU(),
      nn.Tanh(),
    )
    self.obs_normalizer = HoraRunningMeanStd(self.obs_dim)
    for module in self.modules():
      if isinstance(module, nn.Linear):
        module.reset_parameters()
        nn.init.zeros_(module.bias)

  def _get_latent_dim(self) -> int:
    return self.obs_dim + 8

  def get_latent(self, obs, masks=None, hidden_state=None):
    latent = super().get_latent(obs, masks, hidden_state)
    return torch.cat((latent, self.env_encoder(obs["privileged"])), dim=-1)

  def update_normalization(self, obs):
    # HORA updates input statistics once at each optimization minibatch.
    pass
