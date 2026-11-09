"""L20 shared official-size sphere/cylinder/box configuration."""

from mjlab.envs.mdp.dr import pd_gains
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from ... import mdp
from ...rotation_env_cfg import l20_rotation_env_cfg


def official_mixed_env_cfg(play=False):
  cfg = l20_rotation_env_cfg(play)
  cfg.events.update(
    {
      "physics": EventTermCfg(func=mdp.configure_domains, mode="startup"),
      "disturbance": EventTermCfg(func=mdp.disturb_object, mode="step"),
      "gains": EventTermCfg(
        func=pd_gains,
        mode="reset",
        params={
          "asset_cfg": SceneEntityCfg("hand"),
          "kp_range": (2.9 / 3, 3.1 / 3),
          "kd_range": (0.9, 1.1),
          "operation": "scale",
        },
      ),
      "tracking": EventTermCfg(func=mdp.track_episodes, mode="step"),
    }
  )
  cfg.observations["actor"].terms["frame"].params = {"noise_rad": 0.02}
  return cfg
