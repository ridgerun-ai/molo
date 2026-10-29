"""Exercise MAX export registration and the molo export command.

Run with: uv run --no-sync python tests/test_max_export.py
"""

import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from ultralytics.engine import exporter

from molo.backends.max.pipelines.architectures.yolo11.artifact import load_artifact
from molo.models.yolo.export import install_max_exporter

original = exporter.export_formats()
install_max_exporter()
install_max_exporter()
updated = exporter.export_formats()
assert updated["Argument"].count("max") == 1
for key, values in original.items():
    assert updated[key][:-1] == tuple(values)

with TemporaryDirectory() as temporary:
    source = Path(temporary) / "model.pt"
    shutil.copyfile("yolo11n.pt", source)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "molo",
            "export",
            f"model={source}",
            "format=max",
            "imgsz=64",
        ],
        check=True,
    )
    artifact = source.with_suffix(".max")
    source.unlink()
    weights, names = load_artifact(artifact)
    assert len(names) == 80
    assert "layer0.conv.weight" in weights
    assert not source.exists()
    print("PASS MAX export produces standalone artifact", flush=True)
