"""RSL-RL PPO configuration for L20 HORA Stage 1 rotation."""

from dataclasses import dataclass
from math import ceil

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@dataclass
class HoraPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
  class_name: str = "mjlab.tasks.rotation.rl.ppo:HoraPPO"
  bounds_loss_coef: float = 1e-4


def l20_rotation_ppo_cfg() -> RslRlOnPolicyRunnerCfg:
  model = "mjlab.tasks.rotation.rl.models:HoraModel"
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      class_name=model,
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "log",
      },
    ),
    critic=RslRlModelCfg(
      class_name=model,
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    obs_groups={"actor": ("actor",), "critic": ("actor",)},
    algorithm=HoraPpoAlgorithmCfg(
      learning_rate=0.005,
      num_learning_epochs=5,
      num_mini_batches=4,
      entropy_coef=0.0,
      value_loss_coef=2.0,  # HORA: 0.5 * critic_coef(4).
      clip_param=0.2,
      gamma=0.99,
      lam=0.95,
      desired_kl=0.02,
    ),
    num_steps_per_env=8,
    max_iterations=ceil(1_500_000_000 / (1539 * 8)),
    experiment_name="l20_hora_stage1_official_mixed",
    save_interval=100,
    logger="tensorboard",
    upload_model=False,
  )
