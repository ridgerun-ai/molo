"""Verify molo's Python API with MAX and standard Ultralytics models.

Run with: uv run --no-sync python tests/integration/models/yolo/test_python_api.py
"""

import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from ultralytics import YOLO as UltralyticsYOLO
from molo import YOLO

assert YOLO is UltralyticsYOLO
with TemporaryDirectory() as tmp:
    source = Path(tmp) / "model.pt"
    shutil.copyfile("yolo11n.pt", source)
    model = YOLO(source)
    export = Path(model.export(format="max", imgsz=64))
    assert export == source.with_suffix(".max") and export.is_file()
    torchscript = Path(model.export(format="torchscript", imgsz=64))
    assert torchscript.is_file()
    image = np.random.default_rng(41).integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    pt = model.predict(source=image, imgsz=64, verbose=False)[0]
    source.unlink()
    max_model = YOLO(export, task="detect")
    result = max_model.predict(source=image, imgsz=64, verbose=False)[0]
    assert result.boxes is not None and pt.boxes is not None
    np.testing.assert_allclose(
        np.asarray(result.boxes.data),
        np.asarray(pt.boxes.data),
        atol=0.003,
        rtol=1e-4,
    )
    print(
        "PASS Python MAX export and standalone prediction; PyTorch prediction and TorchScript export",
        flush=True,
    )
