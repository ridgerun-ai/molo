"""YOLO11 model sizes and their scaling parameters."""

from dataclasses import dataclass
from typing import Literal

YOLO11Size = Literal["n", "s", "m", "l", "x"]


@dataclass(frozen=True)
class YOLO11Scale:
    """Architecture scaling parameters for a YOLO11 model size."""

    depth: float
    width: float
    max_channels: int
    c3k: bool


YOLO11_SIZES: dict[YOLO11Size, YOLO11Scale] = {
    "n": YOLO11Scale(depth=0.5, width=0.25, max_channels=1024, c3k=False),
    "s": YOLO11Scale(depth=0.5, width=0.5, max_channels=1024, c3k=False),
    "m": YOLO11Scale(depth=0.5, width=1.0, max_channels=512, c3k=True),
    "l": YOLO11Scale(depth=1.0, width=1.0, max_channels=512, c3k=True),
    "x": YOLO11Scale(depth=1.0, width=1.5, max_channels=512, c3k=True),
}
