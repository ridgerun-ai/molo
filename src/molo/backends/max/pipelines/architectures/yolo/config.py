"""Shared configuration for MAX YOLO detection models."""

from dataclasses import dataclass
from typing import Literal

from max.graph import DeviceRef


YOLOSize = Literal["n", "s", "m", "l", "x"]


@dataclass(frozen=True)
class ImageShape:
    """Image dimensions and batch size used to build and cache a MAX graph."""

    height: int
    width: int
    batch_size: int = 1


@dataclass(frozen=True)
class YOLOConfig:
    """Model size, class count, and MAX graph settings."""

    size: YOLOSize = "n"
    nc: int = 80
    input_size: int = 640
    reg_max: int = 16
    device: DeviceRef = DeviceRef.CPU()
