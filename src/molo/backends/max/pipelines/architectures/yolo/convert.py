"""Convert Ultralytics YOLO checkpoints to MAX weights."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ultralytics.nn.tasks import DetectionModel

import numpy as np

from .artifact import save_artifact


def assign_module(target: dict[str, np.ndarray], prefix: str, module) -> None:
    """Copy a PyTorch module's parameters and running statistics."""
    for name, value in module.state_dict().items():
        if name.endswith("num_batches_tracked") or name == "dfl.conv.weight":
            continue
        target[f"{prefix}.{name}"] = value.detach().cpu().numpy().astype(np.float32)

    # Ultralytics fuses Conv + BatchNorm before calling format exporters.
    # Represent fused convolutions with an identity BatchNorm and their bias.
    from ultralytics.nn.modules.conv import Conv

    for name, child in module.named_modules():
        if isinstance(child, Conv) and not hasattr(child, "bn"):
            root = f"{prefix}.{name}" if name else prefix
            channels = child.conv.out_channels
            bias = target.pop(f"{root}.conv.bias")
            target[f"{root}.bn.running_mean"] = np.zeros(channels, dtype=np.float32)
            target[f"{root}.bn.running_var"] = np.ones(channels, dtype=np.float32)
            target[f"{root}.bn.weight"] = np.full(
                channels, np.sqrt(1.001), dtype=np.float32
            )
            target[f"{root}.bn.bias"] = bias


def convert_model(
    model: DetectionModel, destination: str | Path, *, model_id: str
) -> None:
    """Export an already-loaded Ultralytics model to a MAX artifact."""
    from .. import get_architecture

    architecture = get_architecture(model_id)
    artifact = architecture.convert_model(model)
    save_artifact(destination, artifact)
