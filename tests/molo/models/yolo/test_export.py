# Copyright (C) 2026 RidgeRun, LLC (http://www.ridgerun.com)
# SPDX-License-Identifier: AGPL-3.0-only

"""Tests for YOLO export integration."""

import subprocess
import sys
import unittest
from unittest.mock import patch
from typing import Any, cast

import numpy as np
import torch
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
            self.assertIn("model.0.conv.weight", result.weights)
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

    def test_loaded_model_export(self):
        for source in ("yolo11n.yaml", "yolov8n.pt"):
            with self.subTest(source=source):
                self.check_loaded_model_export(source)

    def check_loaded_model_export(self, source):
        from molo.models.yolo.backend import MaxBackend

        model = YOLO(source)
        reference = cast(Any, model.model)
        reference.pt_path = "in_memory.pt"
        reference.eval()
        with torch.no_grad():
            reference.model[0].conv.weight.mul_(0.5)
        artifact = Path("in_memory.max")
        if artifact.exists():
            raise FileExistsError(f"Refusing to overwrite {artifact}")
        x = np.random.default_rng(12).random((1, 3, 64, 64), dtype=np.float32)
        try:
            with patch(
                "ultralytics.YOLO",
                side_effect=AssertionError("Exporter must not reload a checkpoint"),
            ):
                result = model.export(format="max", imgsz=64)
            self.assertEqual(Path(result), artifact)
            backend = MaxBackend(str(artifact), device=torch.device("cpu"))
            with torch.inference_mode():
                expected = reference(torch.from_numpy(x))[0].numpy()
                actual = backend.forward(torch.from_numpy(x)).numpy()
            np.testing.assert_allclose(actual, expected, atol=0.003, rtol=1e-4)
        finally:
            artifact.unlink(missing_ok=True)

    def test_yolo26_rejects_non_end_to_end_export(self):
        source = YOLO("yolo26n.yaml")
        with (
            patch(
                "molo.backends.max.pipelines.architectures.yolo.convert.save_artifact"
            ) as save_artifact,
            self.assertRaisesRegex(ValueError, "requires end2end=True"),
        ):
            source.export(format="max", imgsz=64, end2end=False)
        save_artifact.assert_not_called()

    def test_yolo26_rejects_non_default_selection(self):
        for options, message in (
            ({"max_det": 10}, "requires max_det=300"),
            ({"max_det": 400}, "requires max_det=300"),
            ({"agnostic_nms": True}, "requires agnostic_nms=False"),
        ):
            with self.subTest(options=options):
                source = YOLO("yolo26n.yaml")
                with (
                    patch(
                        "molo.backends.max.pipelines.architectures.yolo.convert.save_artifact"
                    ) as save_artifact,
                    self.assertRaisesRegex(ValueError, message),
                ):
                    source.export(format="max", imgsz=64, **options)
                save_artifact.assert_not_called()

    def test_yolo26_default_export_with_small_image(self):
        source = YOLO("yolo26n.yaml")
        with patch(
            "molo.backends.max.pipelines.architectures.yolo.convert.save_artifact"
        ) as save_artifact:
            source.export(format="max", imgsz=64)
        artifact = save_artifact.call_args.args[1]
        self.assertEqual(artifact.model_id, "yolo26n")
        self.assertIn("model.23.one2one_cv2.0.2.weight", artifact.weights)

    def test_yolo26_export_and_prediction(self):
        artifact = Path("yolo26_verified.max")
        if artifact.exists():
            raise FileExistsError(f"Refusing to overwrite {artifact}")
        try:
            source = YOLO("yolo26n.pt")
            reference = cast(Any, source.model)
            reference.pt_path = "yolo26_verified.pt"
            output = source.export(format="max", imgsz=64)
            self.assertEqual(Path(output), artifact)
            exported = load_artifact(artifact)
            self.assertEqual(exported.model_id, "yolo26n")
            self.assertIn("model.23.one2one_cv2.0.2.weight", exported.weights)
            self.assertNotIn("model.23.cv2.0.2.weight", exported.weights)
            images = list(
                np.random.default_rng(24).integers(
                    0, 256, (2, 64, 64, 3), dtype=np.uint8
                )
            )
            expected = source.predict(
                images, batch=2, imgsz=64, conf=0.0, verbose=False
            )
            actual = YOLO(artifact, task="detect").predict(
                images, batch=2, imgsz=64, conf=0.0, verbose=False
            )
            self.assertEqual(len(actual), len(images))
            for result, reference_result in zip(actual, expected):
                assert result.boxes is not None and reference_result.boxes is not None
                self.assertGreater(len(reference_result.boxes), 0)
                np.testing.assert_allclose(
                    np.asarray(result.boxes.data),
                    np.asarray(reference_result.boxes.data),
                    atol=0.003,
                    rtol=1e-4,
                )
        finally:
            artifact.unlink(missing_ok=True)
