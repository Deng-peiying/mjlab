"""Object displacement termination."""

from __future__ import annotations

from typing import Any

import torch


def dropped(env: Any) -> torch.Tensor:
  pos = env.scene["object"].data.root_link_pos_w
  return (pos - env.hora.centers).norm(dim=-1) > 0.03
