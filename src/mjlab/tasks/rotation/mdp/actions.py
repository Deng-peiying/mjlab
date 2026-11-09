"""Incremental position actions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, cast

import torch

from mjlab.managers.action_manager import ActionTerm, ActionTermCfg

from .events import SharedReset


class RotationStateOwner(Protocol):
  hora: SharedReset


@dataclass(kw_only=True)
class IncrementalPositionCfg(ActionTermCfg):
  def build(self, env: Any) -> IncrementalPosition:
    return IncrementalPosition(self, env)


class IncrementalPosition(ActionTerm):
  def __init__(self, cfg: IncrementalPositionCfg, env: Any):
    super().__init__(cfg, env)
    self.raw = torch.zeros((self.num_envs, 16), device=self.device)

  @property
  def action_dim(self) -> int:
    return 16

  @property
  def raw_action(self) -> torch.Tensor:
    return self.raw

  def process_actions(self, actions: torch.Tensor):
    self.raw[:] = actions
    state = cast(RotationStateOwner, self._env).hora
    # Snapshot once per control step, before its physics substeps and auto-reset.
    state.previous_pose[:] = self._env.scene["object"].data.root_link_pose_w
    state.previous_joints[:] = self._entity.data.joint_pos[:, state.ids]
    limits = self._entity.data.joint_pos_limits[:, state.ids]
    state.targets[:] = torch.clamp(
      state.targets + actions.clamp(-1, 1) / 24, min=limits[..., 0], max=limits[..., 1]
    )

  def apply_actions(self):
    state = cast(RotationStateOwner, self._env).hora
    self._entity.set_joint_position_target(state.targets, state.ids)

  def reset(self, env_ids=None):
    self.raw[env_ids] = 0
