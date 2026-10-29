"""Read and write self-contained MOLO MAX weight artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .model_config import YOLO11_SIZES, YOLO11Size
from typing import cast

_METADATA_KEY = "__molo_metadata__"


def save_artifact(
    path: str | Path, weights: dict[str, np.ndarray], names: dict[int, str], size: str
) -> None:
    """Write weights and class names to a single .max file."""
    path = Path(path)
    metadata = {
        "model_id": f"yolo11{size}",
        "names": {str(k): v for k, v in names.items()},
    }
    payload = json.dumps(metadata, ensure_ascii=False).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as output:
        np.savez(
            output,
            **weights,  # ty: ignore[invalid-argument-type]
            **{_METADATA_KEY: np.frombuffer(payload, dtype=np.uint8)},
        )


def load_artifact(
    path: str | Path,
) -> tuple[dict[str, np.ndarray], dict[int, str], YOLO11Size]:
    """Load a MOLO .max file without reading any source checkpoint."""
    path = Path(path)
    try:
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(archive[_METADATA_KEY].tobytes().decode("utf-8"))
            model_id = metadata.get("model_id")
            if model_id not in {f"yolo11{size}" for size in YOLO11_SIZES}:
                raise ValueError(f"Unsupported model identifier: {model_id!r}")
            size = cast(YOLO11Size, model_id.removeprefix("yolo11"))
            names = {int(k): str(v) for k, v in metadata["names"].items()}
            weights = {
                key: archive[key] for key in archive.files if key != _METADATA_KEY
            }
    except (
        OSError,
        KeyError,
        TypeError,
        UnicodeDecodeError,
        json.JSONDecodeError,
    ) as exc:
        raise ValueError(f"Invalid MOLO .max artifact: {path}") from exc
    return weights, names, size
