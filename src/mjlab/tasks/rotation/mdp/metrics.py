"""Per-domain completed-episode statistics."""

from __future__ import annotations

import csv
from pathlib import Path

import torch

from mjlab.utils.lab_api.math import axis_angle_from_quat, quat_conjugate, quat_mul

from .constants import (
  SHAPES,
)


def flush_metrics(env):
  state = env.hora
  if state.metric_path is None:
    return
  groups = [f"shape/{s}" for s in SHAPES]
  groups += [f"size/{s}/{p / 100:.2f}" for s in SHAPES for p in range(70, 87, 2)]
  groups += [f"cylinder_ratio/{p / 100:.2f}" for p in range(80, 121, 5)]
  data = state.aggregates.detach().cpu().numpy()
  path = Path(state.metric_path)
  fresh = not path.exists()
  with path.open("a", newline="") as stream:
    writer = csv.writer(stream)
    if fresh:
      writer.writerow(
        [
          "control_step",
          "group",
          "episodes",
          "drops",
          "net_rad_sum",
          "hold_s_sum",
          "absolute_rad_sum",
          "reward_sum",
        ]
      )
    for group, values in zip(groups, data, strict=True):
      if values[0]:
        writer.writerow([state.metric_ticks, group, *values.tolist()])
  state.aggregates.zero_()


def track_episodes(env, env_ids):
  state = env.hora
  pose = env.scene["object"].data.root_link_quat_w
  delta = -axis_angle_from_quat(
    quat_mul(pose, quat_conjugate(state.previous_pose[:, 3:]))
  )[:, 0]
  state.net += delta
  state.absolute += delta.abs()
  state.age += env.step_dt
  state.episode_reward += env.reward_buf
  done = env.reset_terminated | env.reset_time_outs
  value = (
    torch.stack(
      (
        torch.ones_like(delta),
        env.reset_terminated.float(),
        state.net,
        state.age,
        state.absolute,
        state.episode_reward,
      ),
      -1,
    )
    * done[:, None]
  )
  shape = state.shape_table[state.domain_bank]
  size = ((state.scale_table[state.domain_bank] - 0.70) / 0.02).round().long()
  ratio = (
    ((state.ratio_table[state.domain_bank] - 0.8) / 0.05).round().long().clamp(0, 8)
  )
  state.aggregates.index_add_(0, shape, value)
  state.aggregates.index_add_(0, 3 + shape * 9 + size, value)
  state.aggregates.index_add_(0, 30 + ratio, value * (shape == 1)[:, None])
  state.metric_ticks += 1
  if state.metric_ticks % 100 == 0:
    flush_metrics(env)
