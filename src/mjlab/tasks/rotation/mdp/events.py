"""Cached state resets and domain randomization."""

from __future__ import annotations

import json
from typing import Any

import mujoco
import numpy as np
import torch

from mjlab.envs.mdp.dr.geom import _recompute_geom_bounds
from mjlab.managers.event_manager import RecomputeLevel, requires_model_fields
from mjlab.managers.manager_base import ManagerTermBase

from .constants import ACTIVE, ASSETS, DIRECTORY, MODEL, SHAPES


class SharedReset(ManagerTermBase):
  def cache_names(self):
    self.manifest = json.loads((DIRECTORY / "manifest.json").read_text())
    return tuple((str(DIRECTORY / b["file"]) for b in self.manifest["banks"]))

  def check_cache(self, name, cache):
    shape, scale, ratio = (
      str(cache["shape"]),
      float(cache["scale"]),
      float(cache["ratio"]),
    )
    radius = 0.04 * scale
    expected = {
      "sphere": [radius, 0, 0],
      "cylinder": [radius, radius * ratio, 0],
      "box": [radius] * 3,
    }[shape]
    if not np.allclose(cache["geom_size"], expected):
      raise ValueError(f"Cache geometry mismatch: {name}")

  def __init__(self, cfg: Any, env: Any):
    super().__init__(env)
    del cfg
    hand = env.scene["hand"]
    reference = mujoco.MjModel.from_xml_path(str(MODEL))
    columns = [int(reference.joint(n).qposadr[0]) for n in hand.joint_names]
    poses, objects, sizes = ([], [], [])
    for name in self.cache_names():
      with np.load(ASSETS / name, allow_pickle=False) as cache:
        if tuple(cache["joint_names"]) != ACTIVE or not len(cache["qpos"]):
          raise ValueError(f"Empty or incompatible grasp cache: {name}")
        self.check_cache(name, cache)
        poses.append(cache["qpos"][:, columns])
        objects.append(cache["qpos"][:, -7:])
        sizes.append(len(cache["qpos"]))
    self.poses, self.objects = (
      torch.as_tensor(np.concatenate(rows), device=self.device, dtype=torch.float32)
      for rows in (poses, objects)
    )
    self.bank_counts = torch.tensor(sizes, device=self.device)
    self.bank_offsets = self.bank_counts.cumsum(0) - self.bank_counts
    self.ids = torch.tensor(
      hand.find_joints(ACTIVE, preserve_order=True)[0],
      device=self.device,
      dtype=torch.long,
    )
    self.initial = torch.zeros((self.num_envs, 16), device=self.device)
    self.targets = torch.zeros_like(self.initial)
    self.centers = torch.zeros((self.num_envs, 3), device=self.device)
    self.previous_pose = torch.zeros((self.num_envs, 7), device=self.device)
    self.previous_joints = torch.zeros_like(self.initial)
    self.sample_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self.forced_samples: torch.Tensor | None = None
    env.hora = self
    banks = self.manifest["banks"]
    self.shape_table = torch.tensor(
      [SHAPES.index(b["shape"]) for b in banks], device=self.device
    )
    self.scale_table = torch.tensor([b["scale"] for b in banks], device=self.device)
    self.ratio_table = torch.tensor([b["ratio"] for b in banks], device=self.device)
    world = torch.arange(self.num_envs, device=self.device)
    shape, size, ratio = (world % 3, world // 3 % 9, world // 27 % 9)
    self.domain_bank = torch.where(
      shape == 0, size, torch.where(shape == 1, 9 + size * 9 + ratio, 90 + size)
    )
    self.net = torch.zeros(self.num_envs, device=self.device)
    self.absolute = torch.zeros_like(self.net)
    self.age = torch.zeros_like(self.net)
    self.episode_reward = torch.zeros_like(self.net)
    self.aggregates = torch.zeros((39, 6), device=self.device)
    self.metric_path = None
    self.metric_ticks = 0

  def bank_indices(self, env_ids):
    return self.domain_bank[env_ids]

  def __call__(self, env: Any, env_ids: torch.Tensor | None):
    if env_ids is None:
      env_ids = torch.arange(self.num_envs, device=self.device)
    samples = self.forced_samples
    if samples is None:
      bank = self.bank_indices(env_ids)
      samples = (
        self.bank_offsets[bank]
        + (torch.rand(len(env_ids), device=self.device) * self.bank_counts[bank]).long()
      )
    self.forced_samples = None
    self.sample_ids[env_ids] = samples
    pose = self.poses[samples]
    env.scene["hand"].write_joint_state_to_sim(
      pose, torch.zeros_like(pose), env_ids=env_ids
    )
    self.initial[env_ids] = pose[:, self.ids]
    self.targets[env_ids] = self.initial[env_ids]
    env.scene["hand"].set_joint_position_target(
      self.targets[env_ids], self.ids, env_ids
    )
    root = torch.zeros((len(env_ids), 13), device=self.device)
    root[:, :7] = self.objects[samples]
    root[:, :3] += env.scene.env_origins[env_ids]
    env.scene["object"].write_root_state_to_sim(root, env_ids)
    self.centers[env_ids] = root[:, :3]
    for value in (self.net, self.absolute, self.age, self.episode_reward):
      value[env_ids] = 0


def disturb_object(env, env_ids):
  obj = env.scene["object"]
  force = obj.data.body_external_force
  mass = env.sim.model.body_mass[:, obj.indexing.body_ids[0]]
  replace = torch.rand(env.num_envs, 1, 1, device=env.device) < 0.25
  force = torch.where(
    replace,
    torch.randn_like(force) * (2 * mass[:, None, None]),
    force * 0.9 ** (env.step_dt / 0.08),
  )
  obj.write_external_wrench_to_sim(force, torch.zeros_like(force))


@requires_model_fields(
  "geom_size",
  "geom_rbound",
  "geom_aabb",
  "geom_friction",
  "body_mass",
  "body_ipos",
  "body_inertia",
  "body_iquat",
  recompute=RecomputeLevel.set_const,
)
def configure_domains(env, env_ids, full_dr=True, jitter=True):
  state, model, obj = (env.hora, env.sim.model, env.scene["object"])
  ids = torch.arange(env.num_envs, device=env.device)
  shape = state.shape_table[state.domain_bank]
  scale = state.scale_table[state.domain_bank].clone()
  ratio = state.ratio_table[state.domain_bank]
  if jitter:
    scale += torch.empty_like(scale).uniform_(-0.025, 0.025)
  radius = 0.04 * scale
  sizes = torch.stack(
    (
      radius,
      torch.where(
        shape == 0, radius * 0, torch.where(shape == 1, radius * ratio, radius)
      ),
      torch.where(shape == 2, radius, radius * 0),
    ),
    -1,
  )
  geoms = obj.indexing.geom_ids.long()
  body = obj.indexing.body_ids[0]
  model.geom_size[ids[:, None], geoms] = 1e-06
  model.geom_size[ids, geoms[shape]] = sizes
  _recompute_geom_bounds(env, ids, geoms)
  mass = (
    torch.empty_like(radius).uniform_(0.01, 0.25)
    if full_dr
    else torch.full_like(radius, 0.05)
  )
  friction = (
    torch.empty_like(radius).uniform_(0.3, 3) if full_dr else torch.ones_like(radius)
  )
  model.body_mass[:, body] = mass
  model.body_ipos[:, body] = (
    radius.new_empty((env.num_envs, 3)).uniform_(-0.01, 0.01) if full_dr else 0
  )
  model.body_iquat[:, body] = radius.new_tensor([1, 0, 0, 0])
  model.geom_friction[:, :, 0] = friction[:, None]
  a, b, c = sizes.square().unbind(-1)
  sphere = (0.4 * a)[:, None].expand(-1, 3)
  radial = a / 4 + b / 3
  cylinder = torch.stack((radial, radial, a / 2), -1)
  box = torch.stack((b + c, a + c, a + b), -1) / 3
  model.body_inertia[:, body] = mass[:, None] * torch.where(
    (shape == 0)[:, None], sphere, torch.where((shape == 1)[:, None], cylinder, box)
  )
