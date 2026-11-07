"""Map Ultralytics YOLO11 weights to the MAX model's weight names."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .model_config import YOLO11_SIZES


def _assign_module(target: dict[str, np.ndarray], prefix: str, module) -> None:
    """Copy an unfused PyTorch module's parameters and running statistics."""
    for name, value in module.state_dict().items():
        if name.endswith("num_batches_tracked") or name == "dfl.conv.weight":
            continue
        target[f"{prefix}.{name}"] = value.detach().cpu().numpy().astype(np.float32)


def _assign_conv(
    target: dict[str, np.ndarray], prefix: str, module, *, max_name: str | None = None
) -> None:
    _assign_module(target, prefix, module)


def _assign_c3k2(target: dict[str, np.ndarray], prefix: str, block) -> None:
    _assign_module(target, prefix, block)


def _assign_sppf(target: dict[str, np.ndarray], prefix: str, block) -> None:
    _assign_module(target, prefix, block)


def _assign_c2psa(target: dict[str, np.ndarray], prefix: str, block) -> None:
    _assign_module(target, prefix, block)


def _assign_detect(target: dict[str, np.ndarray], prefix: str, block) -> None:
    _assign_module(target, prefix, block)


def convert_checkpoint(source: str | Path, destination: str | Path) -> None:
    from ultralytics import YOLO

    from .artifact import save_artifact

    yolo = YOLO(str(source))
    model = yolo.model.model  # ty: ignore[unresolved-attribute]
    names = dict(yolo.names)
    model_id = Path(yolo.model.yaml.get("yaml_file", "")).stem  # ty: ignore[unresolved-attribute]
    if model_id not in {f"yolo11{size}" for size in YOLO11_SIZES}:
        raise ValueError(f"Unsupported model identifier: {model_id!r}")
    size = model_id.removeprefix("yolo11")
    target: dict[str, np.ndarray] = {}
    for idx in (0, 1, 3, 5, 7, 17, 20):
        _assign_conv(target, f"layer{idx}", model[idx])
    for idx in (2, 4, 6, 8, 13, 16, 19, 22):
        _assign_c3k2(target, f"layer{idx}", model[idx])
    _assign_sppf(target, "layer9", model[9])
    _assign_c2psa(target, "layer10", model[10])
    _assign_detect(target, "detect", model[23])
    save_artifact(destination, target, names, size)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Ultralytics YOLO11 weights for MAX."
    )
    parser.add_argument("source", help="Path to an Ultralytics YOLO11 checkpoint")
    parser.add_argument("destination", help="Path to the output .max file")
    args = parser.parse_args()
    convert_checkpoint(args.source, args.destination)


if __name__ == "__main__":
    main()
