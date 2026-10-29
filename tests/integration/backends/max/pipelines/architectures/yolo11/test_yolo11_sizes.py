"""Compare all YOLO11 sizes using official checkpoints.

Run with: uv run --no-sync python tests/integration/backends/max/pipelines/architectures/yolo11/test_yolo11_sizes.py
"""

import gc
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO

from molo.backends.max.pipelines.architectures.yolo11.convert import convert_checkpoint
from molo.models.yolo.backend import MaxBackend

x = np.random.default_rng(24).random((1, 3, 64, 64), dtype=np.float32)
for size in ("n", "s", "m", "l", "x"):
    source = Path(f"yolo11{size}.pt")
    artifact = source.with_suffix(".max")
    if artifact.exists():
        raise FileExistsError(f"Refusing to overwrite {artifact}")
    reference = YOLO(str(source)).model.eval()  # ty: ignore[unresolved-attribute]
    convert_checkpoint(source, artifact)
    backend = MaxBackend(str(artifact), device=torch.device("cpu"))
    assert backend.config is not None
    assert backend.config.size == size
    with torch.inference_mode():
        expected = reference(torch.from_numpy(x))[0].numpy()
        actual = backend.forward(torch.from_numpy(x)).numpy()
    np.testing.assert_allclose(actual, expected, atol=0.003, rtol=1e-4)
    print(
        f"PASS YOLO11{size} max_abs={np.max(np.abs(actual - expected)):.6g}", flush=True
    )
    del backend, reference, expected, actual
    gc.collect()
