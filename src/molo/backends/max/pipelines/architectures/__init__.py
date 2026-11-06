"""Register and construct supported model architectures."""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from ultralytics.nn.tasks import DetectionModel
    from max.graph import Graph


from .yolo11.model import (
    convert_model as convert_yolo11,
    identify_model as identify_yolo11,
)
from .yolo.config import ImageShape, YOLOConfig
from .yolo.artifact import YOLOArtifact
from .yolo11.model import build_graph as build_yolo11_graph
from .yolo11.model import YOLO11_SIZES
from .yolov8.model import (
    convert_model as convert_yolov8,
    identify_model as identify_yolov8,
)
from .yolov8.model import build_graph as build_yolov8_graph
from .yolov8.model import YOLOV8_SIZES


@dataclass(frozen=True)
class Architecture:
    """Operations supplied by an architecture package."""

    prefix: str
    sizes: Collection[str]
    identify_model: Callable[[DetectionModel], YOLOConfig]
    build_graph: Callable[[YOLOConfig, ImageShape | None], Graph]
    convert_model: Callable[[DetectionModel], YOLOArtifact]


_ARCHITECTURES: dict[str, Architecture] = {}


def register_architecture(architecture: Architecture) -> None:
    """Register a model family once, including its supported sizes."""
    for size in architecture.sizes:
        _ARCHITECTURES[f"{architecture.prefix}{size}"] = architecture


def get_architecture(model_id: str) -> Architecture:
    """Resolve an explicitly registered model identifier."""
    try:
        return _ARCHITECTURES[model_id]
    except KeyError:
        raise ValueError(f"Unsupported model identifier: {model_id!r}") from None


register_architecture(
    Architecture(
        prefix="yolo11",
        identify_model=identify_yolo11,
        sizes=YOLO11_SIZES,
        build_graph=build_yolo11_graph,
        convert_model=convert_yolo11,
    ),
)
register_architecture(
    Architecture(
        prefix="yolov8",
        identify_model=identify_yolov8,
        sizes=YOLOV8_SIZES,
        build_graph=build_yolov8_graph,
        convert_model=convert_yolov8,
    ),
)
