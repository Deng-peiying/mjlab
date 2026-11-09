"""Hand and object MuJoCo specifications."""

from __future__ import annotations

import mujoco

from ..mdp.constants import ACTIVE, MODEL


def hand_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(MODEL))
  for geom in spec.geoms:
    geom.contype, geom.conaffinity = (1, 2)
    geom.friction = [1, 0.005, 0.0001]
  for name in ACTIVE:
    joint = spec.joint(name)
    actuator = spec.add_actuator(
      name=name,
      target=name,
      trntype=mujoco.mjtTrn.mjTRN_JOINT,
      ctrllimited=True,
      ctrlrange=joint.range.tolist(),
    )
    actuator.set_to_position(kp=20, kv=1)
  for joint in spec.joints:
    joint.armature, joint.damping = (0.01, [0.1, 0, 0])
  for equality in spec.equalities:
    equality.solref = [0.002, 1]
  return spec


def official_object_spec():
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="object")
  body.add_freejoint(name="object_free")
  for shape, size in (
    ("sphere", [0.032, 0, 0]),
    ("cylinder", [0.032, 0.032, 0]),
    ("box", [0.032] * 3),
  ):
    body.add_geom(
      name=shape,
      type=getattr(mujoco.mjtGeom, "mjGEOM_" + shape.upper()),
      size=size,
      mass=0.05 / 3,
      contype=2,
      conaffinity=1,
      friction=[1, 0.005, 0.0001],
      rgba=[0.8, 0.3, 0.1, 1],
    )
  return spec
