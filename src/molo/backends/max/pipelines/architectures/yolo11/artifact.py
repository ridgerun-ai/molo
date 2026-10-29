"""Read and write self-contained MOLO MAX weight artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import numpy as np

from .model_config import YOLO11_SIZES, YOLO11Size

_METADATA_KEY = "__molo_metadata__"
_FORMAT = "molo.yolo11.weights"


def save_artifact(
    path: str | Path, weights: dict[str, np.ndarray], names: dict[int, str], size: str
) -> None:
    """Write weights and class names to a single .max file."""
    path = Path(path)
    if path.suffix.lower() != ".max":
        raise ValueError("MAX artifact must have a .max extension")
    if size not in YOLO11_SIZES:
        raise ValueError(f"Unsupported YOLO11 size: {size!r}")
    metadata = {
        "format": _FORMAT,
        "size": size,
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
    if path.suffix.lower() != ".max":
        raise ValueError("Expected a .max artifact")
    try:
        with np.load(path, allow_pickle=False) as archive:
            if _METADATA_KEY not in archive:
                raise ValueError(
                    "Not a MOLO .max artifact: missing metadata; convert a .pt checkpoint first"
                )
            metadata = json.loads(archive[_METADATA_KEY].tobytes().decode("utf-8"))
            if metadata.get("format") != _FORMAT:
                raise ValueError("Unsupported MOLO .max artifact format")
            size = metadata.get("size")
            if size not in YOLO11_SIZES:
                raise ValueError(f"Unsupported YOLO11 size in artifact: {size!r}")
            names = {int(k): str(v) for k, v in metadata["names"].items()}
            if not names or sorted(names) != list(range(len(names))):
                raise ValueError("Invalid class names in MOLO .max artifact")
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
    return weights, names, cast(YOLO11Size, size)
