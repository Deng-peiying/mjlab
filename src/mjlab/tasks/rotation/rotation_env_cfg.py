"""Common L20 rotation environment configuration."""

from mjlab.actuator.xml_actuator import XmlActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import time_out
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.viewer import ViewerConfig

from . import mdp
from .assets.specs import hand_spec, official_object_spec


def l20_rotation_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      num_envs=3 if play else 1539,
      env_spacing=0.5,
      entities={
        "hand": EntityCfg(
          spec_fn=hand_spec,
          articulation=EntityArticulationInfoCfg(
            actuators=(XmlActuatorCfg(target_names_expr=mdp.ACTIVE),)
          ),
        ),
        "object": EntityCfg(spec_fn=official_object_spec),
      },
    ),
    actions={"position": mdp.IncrementalPositionCfg(entity_name="hand")},
    events={"grasp": EventTermCfg(func=mdp.SharedReset, mode="reset")},
    observations={
      "actor": ObservationGroupCfg(
        {"frame": ObservationTermCfg(func=mdp.proprioception, history_length=3)},
        enable_corruption=not play,
      ),
      "privileged": ObservationGroupCfg(
        {"object": ObservationTermCfg(func=mdp.properties)}
      ),
    },
    rewards={
      "rotation": RewardTermCfg(func=mdp.rotation, weight=1.0),
      "linear_velocity": RewardTermCfg(func=mdp.linear_speed, weight=-0.3),
      "posture": RewardTermCfg(func=mdp.posture, weight=-0.3),
      "torque": RewardTermCfg(func=mdp.torque, weight=-0.1),
      "work": RewardTermCfg(func=mdp.work, weight=-2.0),
    },
    terminations={
      "dropped": TerminationTermCfg(func=mdp.dropped),
      "timeout": TerminationTermCfg(func=time_out, time_out=True),
    },
    sim=SimulationCfg(
      nconmax=256,
      njmax=2048,
      mujoco=MujocoCfg(timestep=0.002, integrator="euler", gravity=(-9.81, 0, 0)),
    ),
    decimation=25,
    episode_length_s=1e9 if play else 20,
    scale_rewards_by_dt=False,
    viewer=ViewerConfig(
      entity_name="object", body_name="object", distance=0.4, azimuth=155, elevation=-25
    ),
  )
