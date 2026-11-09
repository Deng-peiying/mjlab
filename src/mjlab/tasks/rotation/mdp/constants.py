"""Model paths and object-distribution definitions."""

from pathlib import Path

ASSETS = Path(__file__).resolve().parents[1] / "assets"
MODEL = ASSETS / "l20/l20_right_manual.urdf"
ACTIVE = (
  "thumb_cmc_roll",
  "thumb_cmc_yaw",
  "thumb_cmc_pitch",
  "thumb_mcp",
  *[
    f"{finger}_{axis}"
    for finger in ("index", "middle", "ring", "pinky")
    for axis in ("mcp_roll", "mcp_pitch", "pip")
  ],
)
SHAPES = ("sphere", "cylinder", "box")
DIRECTORY = ASSETS / "official_mixed_v1"
