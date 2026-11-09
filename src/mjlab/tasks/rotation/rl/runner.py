"""HORA runner audit and domain-statistics persistence."""

from __future__ import annotations

from pathlib import Path

from mjlab.rl import MjlabOnPolicyRunner

from ..mdp.constants import DIRECTORY
from ..mdp.metrics import flush_metrics


class OfficialMixedRunner(MjlabOnPolicyRunner):
  def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
    super().__init__(env, train_cfg, log_dir, device)
    if log_dir is not None:
      env.unwrapped.hora.metric_path = Path(log_dir) / "domain_metrics.csv"
      (Path(log_dir) / "input_manifest.json").write_text(
        (DIRECTORY / "manifest.json").read_text()
      )

  def learn(self, *args, **kwargs):
    try:
      return super().learn(*args, **kwargs)
    finally:
      flush_metrics(self.env.unwrapped)
