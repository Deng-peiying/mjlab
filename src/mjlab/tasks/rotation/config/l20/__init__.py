from mjlab.tasks.registry import register_mjlab_task

from ...rl import OfficialMixedRunner
from .env_cfgs import official_mixed_env_cfg
from .rl_cfg import l20_rotation_ppo_cfg

register_mjlab_task(
  task_id="Mjlab-Rotation-L20-Official-Mixed",
  env_cfg=official_mixed_env_cfg(),
  play_env_cfg=official_mixed_env_cfg(play=True),
  rl_cfg=l20_rotation_ppo_cfg(),
  runner_cls=OfficialMixedRunner,
)
