"""Read and write self-contained MOLO MAX weight artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


_METADATA_KEY = "__molo_metadata__"


@dataclass
class YOLOArtifact:
    """Converted MAX weights and their model metadata."""

    model_id: str
    class_names: list[str]
    weights: dict[str, np.ndarray]


def save_artifact(path: str | Path, artifact: YOLOArtifact) -> None:
    """Write weights and class names to a single .max file."""
    path = Path(path)
    metadata = {
        "model_id": artifact.model_id,
        "class_names": artifact.class_names,
    }
    payload = json.dumps(metadata, ensure_ascii=False).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as output:
        np.savez(
            output,
            **artifact.weights,  # ty: ignore[invalid-argument-type]
            **{_METADATA_KEY: np.frombuffer(payload, dtype=np.uint8)},
        )


def load_artifact(path: str | Path) -> YOLOArtifact:
    """Load a MOLO .max file without reading any source checkpoint."""
    path = Path(path)
    try:
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(archive[_METADATA_KEY].tobytes().decode("utf-8"))
            model_id = metadata["model_id"]
            class_names = metadata["class_names"]
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
    return YOLOArtifact(model_id=model_id, class_names=class_names, weights=weights)
