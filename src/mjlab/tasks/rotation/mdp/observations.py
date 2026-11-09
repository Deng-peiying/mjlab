"""Proprioceptive history and physical privileged inputs."""

from __future__ import annotations

from typing import Any

import torch


def proprioception(env: Any, noise_rad: float = 0.0) -> torch.Tensor:
  state = env.hora
  data = env.scene["hand"].data
  q = data.joint_pos[:, state.ids]
  if noise_rad:
    q = q + torch.empty_like(q).uniform_(-noise_rad, noise_rad)
  limits = data.joint_pos_limits[:, state.ids]
  normalized = 2 * (q - limits[..., 0]) / (limits[..., 1] - limits[..., 0]) - 1
  return torch.cat((normalized, state.targets), -1)


def properties(env):
  obj, model, state = (env.scene["object"], env.sim.model, env.hora)
  ids = torch.arange(env.num_envs, device=env.device)
  shape = state.shape_table[state.domain_bank]
  geom, body = (obj.indexing.geom_ids[shape], obj.indexing.body_ids[0])
  return torch.cat(
    (
      obj.data.root_link_pos_w - env.scene.env_origins,
      model.geom_size[ids, geom, 0, None] / 0.04,
      model.body_mass[:, body, None],
      model.geom_friction[ids, geom, 0, None],
      model.body_ipos[:, body],
    ),
    -1,
  )
