"""Exercise MAX export registration and the molo export command.

Run with: uv run --no-sync python tests/integration/models/yolo/test_max_export.py
"""

import subprocess
import sys
from pathlib import Path

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

source = Path("yolo11n.pt")
artifact = source.with_suffix(".max")
if artifact.exists():
    raise FileExistsError(f"Refusing to overwrite {artifact}")
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
weights, names, size = load_artifact(artifact)
assert size == "n"
assert len(names) == 80
assert "layer0.conv.weight" in weights
print("PASS MAX export produces artifact", flush=True)
