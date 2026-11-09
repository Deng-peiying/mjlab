"""HORA-style rotation rewards."""

from __future__ import annotations

from typing import Any

import torch

from mjlab.utils.lab_api.math import axis_angle_from_quat, quat_conjugate, quat_mul


def rotation(env: Any) -> torch.Tensor:
  delta = quat_mul(
    env.scene["object"].data.root_link_quat_w,
    quat_conjugate(env.hora.previous_pose[:, 3:]),
  )
  angular_velocity = axis_angle_from_quat(delta) / env.step_dt
  return (-angular_velocity[:, 0]).clamp(-0.5, 0.5)


def linear_speed(env: Any) -> torch.Tensor:
  displacement = (
    env.scene["object"].data.root_link_pos_w - env.hora.previous_pose[:, :3]
  )
  return (displacement / env.step_dt).abs().sum(-1)


def posture(env: Any) -> torch.Tensor:
  q = env.scene["hand"].data.joint_pos[:, env.hora.ids]
  return (q - env.hora.initial).square().sum(-1)


def torque(env: Any) -> torch.Tensor:
  return env.scene["hand"].data.actuator_force.square().sum(-1)


def work(env: Any) -> torch.Tensor:
  data = env.scene["hand"].data
  velocity = (data.joint_pos[:, env.hora.ids] - env.hora.previous_joints) / env.step_dt
  return (data.qfrc_actuator[:, env.hora.ids] * velocity).sum(-1).square()
