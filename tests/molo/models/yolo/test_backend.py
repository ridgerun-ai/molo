"""Tests for the YOLO MAX backend."""

import unittest

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
