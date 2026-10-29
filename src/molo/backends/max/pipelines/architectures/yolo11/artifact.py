"""Read and write self-contained MOLO MAX weight artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

_METADATA_KEY = "__molo_metadata__"
_FORMAT = "molo.yolo11n.weights.v1"


def save_artifact(
    path: str | Path, weights: dict[str, np.ndarray], names: dict[int, str]
) -> None:
    """Write weights and class names to a single .max file."""
    path = Path(path)
    if path.suffix.lower() != ".max":
        raise ValueError("MAX artifact must have a .max extension")
    metadata = {"format": _FORMAT, "names": {str(k): v for k, v in names.items()}}
    payload = json.dumps(metadata, ensure_ascii=False).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as output:
        np.savez(
            output,
            **weights,  # ty: ignore[invalid-argument-type]
            **{_METADATA_KEY: np.frombuffer(payload, dtype=np.uint8)},
        )


def load_artifact(path: str | Path) -> tuple[dict[str, np.ndarray], dict[int, str]]:
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
    return weights, names
