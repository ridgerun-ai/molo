"""Convert Ultralytics YOLO checkpoints to MAX weights."""

from __future__ import annotations

from pathlib import Path
from copy import deepcopy
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ultralytics.nn.tasks import DetectionModel

import numpy as np

from .artifact import save_artifact


def assign_module(target: dict[str, np.ndarray], prefix: str, module) -> None:
    """Copy parameters, folding YOLO convolution normalization into weights and bias."""
    from ultralytics.nn.modules.conv import Conv
    from ultralytics.utils.torch_utils import fuse_conv_and_bn

    if any(
        isinstance(child, Conv) and hasattr(child, "bn") for child in module.modules()
    ):
        module = deepcopy(module).eval()
        for child in module.modules():
            if isinstance(child, Conv) and hasattr(child, "bn"):
                child.conv = fuse_conv_and_bn(child.conv, child.bn)
                del child.bn
                child.forward = child.forward_fuse
    for name, value in module.state_dict().items():
        if name.endswith("num_batches_tracked"):
            continue
        target[f"{prefix}.{name}" if prefix else name] = (
            value.detach().cpu().numpy().astype(np.float32)
        )


def convert_model(
    model: DetectionModel, destination: str | Path, *, model_id: str
) -> None:
    """Export an already-loaded Ultralytics model to a MAX artifact."""
    from .. import get_architecture

    architecture = get_architecture(model_id)
    artifact = architecture.convert_model(model)
    save_artifact(destination, artifact)
