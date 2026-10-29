"""Compare the MAX backend with Ultralytics using the local YOLO11n checkpoint.

Run with: uv run --no-sync python tests/integration/backends/max/pipelines/architectures/yolo11/test_yolo11_integration.py
"""

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from ultralytics import YOLO
from molo.models.yolo.backend import MaxBackend
from molo.backends.max.pipelines.architectures.yolo11.convert import convert_checkpoint

reference = YOLO("yolo11n.pt").model.eval()  # ty: ignore[unresolved-attribute]
temp_dir = TemporaryDirectory()
artifact = Path(temp_dir.name) / "standalone.max"
convert_checkpoint("yolo11n.pt", artifact)
assert not artifact.with_suffix(".pt").exists()
backend = MaxBackend(str(artifact), device=torch.device("cpu"))
assert backend.names == reference.names
shape = (64, 96)
x = np.random.default_rng(7).random((1, 3, *shape), dtype=np.float32)
with torch.no_grad():
    expected = reference(torch.from_numpy(x))[0].numpy()
actual = backend.forward(torch.from_numpy(x)).numpy()
np.testing.assert_allclose(actual, expected, atol=0.003, rtol=1e-4)
compiled = backend._compiled_models[shape]
actual_again = backend.forward(torch.from_numpy(x)).numpy()
assert backend._compiled_models[shape] is compiled
np.testing.assert_array_equal(actual_again, actual)
print(
    f"PASS backend shape={shape} max_abs={np.max(np.abs(actual - expected)):.6g}; graph cache reused",
    flush=True,
)
