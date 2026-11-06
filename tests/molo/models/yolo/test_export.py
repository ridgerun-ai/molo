"""Tests for YOLO export integration."""

import subprocess
import sys
import unittest

from pathlib import Path

from ultralytics.engine import exporter
from molo import YOLO
from molo.backends.max.pipelines.architectures.yolo.artifact import load_artifact
from molo.models.yolo.export import install_max_exporter


class ExportTest(unittest.TestCase):
    def test_registration(self):
        install_max_exporter()
        formats = exporter.export_formats()
        install_max_exporter()
        self.assertEqual(exporter.export_formats(), formats)
        self.assertEqual(formats["Argument"].count("max"), 1)

    def test_cli_export(self):
        source = Path("yolo11n.pt")
        artifact = source.with_suffix(".max")
        if artifact.exists():
            raise FileExistsError(f"Refusing to overwrite {artifact}")
        try:
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
            result = load_artifact(artifact)
            self.assertEqual(result.model_id, "yolo11n")
            self.assertEqual(len(result.class_names), 80)
            self.assertIn("layer0.conv.weight", result.weights)
        finally:
            artifact.unlink(missing_ok=True)

    def test_python_export(self):
        source = Path("yolo11n.pt")
        outputs = [source.with_suffix(".max"), source.with_suffix(".torchscript")]
        for output in outputs:
            if output.exists():
                raise FileExistsError(f"Refusing to overwrite {output}")
        try:
            model = YOLO(source)
            self.assertEqual(Path(model.export(format="max", imgsz=64)), outputs[0])
            self.assertEqual(
                Path(model.export(format="torchscript", imgsz=64)), outputs[1]
            )
            self.assertTrue(all(output.is_file() for output in outputs))
        finally:
            for output in outputs:
                output.unlink(missing_ok=True)
