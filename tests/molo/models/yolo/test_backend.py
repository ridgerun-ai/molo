"""Tests for the YOLO MAX backend."""

import unittest
from copy import deepcopy
from typing import cast
from ultralytics.nn.tasks import DetectionModel

from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO

from molo.backends.max.pipelines.architectures.yolo.convert import convert_model
from molo.models.yolo.backend import MaxBackend
from molo.backends.max.pipelines.architectures.yolo.config import ImageShape


class BackendTest(unittest.TestCase):
    def test_prediction_and_graph_cache(self):
        source = Path("yolo11n.pt")
        reference = YOLO(str(source)).model.eval()  # ty: ignore[unresolved-attribute]
        artifact = Path("standalone.max")
        if artifact.exists():
            raise FileExistsError(f"Refusing to overwrite {artifact}")
        try:
            convert_model(reference, artifact, model_id="yolo11n")
            self.assertFalse(artifact.with_suffix(".pt").exists())
            backend = MaxBackend(str(artifact), device=torch.device("cpu"))
            self.assertEqual(backend.names, reference.names)
            shape = (64, 96)
            x = np.random.default_rng(7).random((1, 3, *shape), dtype=np.float32)
            with torch.inference_mode():
                expected = reference(torch.from_numpy(x))[0].numpy()
                actual = backend.forward(torch.from_numpy(x)).numpy()
            np.testing.assert_allclose(actual, expected, atol=0.003, rtol=1e-4)
            compiled = backend._compiled_models[ImageShape(height=64, width=96)]
            actual_again = backend.forward(torch.from_numpy(x)).numpy()
            self.assertIs(
                backend._compiled_models[ImageShape(height=64, width=96)], compiled
            )
            np.testing.assert_array_equal(actual_again, actual)
        finally:
            artifact.unlink(missing_ok=True)

    def test_public_prediction(self):
        from molo import YOLO as MoloYOLO

        source = Path("yolo11n.pt")
        artifact = source.with_suffix(".max")
        if artifact.exists():
            raise FileExistsError(f"Refusing to overwrite {artifact}")
        try:
            model = MoloYOLO(source)
            model.export(format="max", imgsz=64)
            image = np.random.default_rng(41).integers(
                0, 256, size=(64, 64, 3), dtype=np.uint8
            )
            expected = model.predict(source=image, imgsz=64, verbose=False)[0]
            actual = MoloYOLO(artifact, task="detect").predict(
                source=image, imgsz=64, verbose=False
            )[0]
            assert actual.boxes is not None and expected.boxes is not None
            np.testing.assert_allclose(
                np.asarray(actual.boxes.data),
                np.asarray(expected.boxes.data),
                atol=0.003,
                rtol=1e-4,
            )
        finally:
            artifact.unlink(missing_ok=True)

    def test_reg_max_inferred_from_weights(self):
        x = np.random.default_rng(57).random((1, 3, 64, 96), dtype=np.float32)
        for family in ("yolo11", "yolov8"):
            yaml = cast(DetectionModel, YOLO(f"{family}n.yaml").model).yaml
            for reg_max in (1, 8, 20):
                with self.subTest(family=family, reg_max=reg_max):
                    cfg = deepcopy(yaml)
                    cfg["reg_max"] = reg_max
                    reference = DetectionModel(cfg, nc=3, verbose=False).eval()
                    artifact = Path(f"{family}_reg_max_{reg_max}.max")
                    if artifact.exists():
                        raise FileExistsError(f"Refusing to overwrite {artifact}")
                    try:
                        convert_model(reference, artifact, model_id=f"{family}n")
                        backend = MaxBackend(str(artifact), device=torch.device("cpu"))
                        assert backend.config is not None
                        self.assertEqual(backend.config.reg_max, reg_max)
                        with torch.inference_mode():
                            expected = reference(torch.from_numpy(x))[0].numpy()
                            actual = backend.forward(torch.from_numpy(x)).numpy()
                        np.testing.assert_allclose(
                            actual, expected, atol=0.003, rtol=1e-4
                        )
                    finally:
                        artifact.unlink(missing_ok=True)
