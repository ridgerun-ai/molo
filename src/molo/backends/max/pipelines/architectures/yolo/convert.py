"""Convert Ultralytics YOLO checkpoints to MAX weights."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ultralytics.nn.tasks import DetectionModel

import numpy as np
import torch

from .artifact import save_artifact


def assign_module(target: dict[str, np.ndarray], prefix: str, module) -> None:
    """Extract prepared FP32 weights without modifying or repairing the module."""
    state = module.state_dict()
    for name, value in state.items():
        if value.is_floating_point() and value.dtype != torch.float32:
            raise ValueError(
                f"MAX conversion requires float32 weights; {name!r} has {value.dtype}"
            )
    if any(
        isinstance(child, torch.nn.modules.batchnorm._BatchNorm)
        for child in module.modules()
    ):
        raise ValueError("MAX conversion requires fused weights")
    for name, value in state.items():
        target[f"{prefix}.{name}" if prefix else name] = value.detach().cpu().numpy()


def convert_model(
    model: DetectionModel, destination: str | Path, *, model_id: str
) -> None:
    """Export an already-loaded Ultralytics model to a MAX artifact."""
    from .. import get_architecture

    architecture = get_architecture(model_id)
    artifact = architecture.convert_model(model)
    save_artifact(destination, artifact)
