"""Tests for the YOLO MAX backend."""

import unittest
from copy import deepcopy
from unittest.mock import patch
from typing import cast
from ultralytics.nn.tasks import DetectionModel

from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO

from molo.backends.max.pipelines.architectures.yolo.convert import convert_model
from molo.models.yolo.backend import MaxBackend
from molo.backends.max.pipelines.architectures.yolo.artifact import YOLOArtifact
from molo.backends.max.pipelines.architectures.yolo.config import ImageShape


class BackendTest(unittest.TestCase):
    def test_prediction_and_graph_cache(self):
        for family in ("yolo11", "yolov8"):
            with self.subTest(family=family):
                reference = (
                    cast(DetectionModel, YOLO(f"{family}n.pt").model)
                    .eval()
                    .fuse(verbose=False)
                )
                artifact = Path(f"{family}_batched.max")
                if artifact.exists():
                    raise FileExistsError(f"Refusing to overwrite {artifact}")
                try:
                    convert_model(reference, artifact, model_id=f"{family}n")
                    self.assertFalse(artifact.with_suffix(".pt").exists())
                    backend = MaxBackend(str(artifact), device=torch.device("cpu"))
                    self.assertEqual(backend.names, reference.names)
                    self.assertFalse(backend._compiled_models)
                    cached = {}
                    for batch_size in (1, 2, 4, 2, 1):
                        with self.subTest(batch_size=batch_size):
                            x = np.random.default_rng(7).random(
                                (batch_size, 3, 64, 96), dtype=np.float32
                            )
                            with torch.inference_mode():
                                expected = reference(torch.from_numpy(x))[0].numpy()
                                actual = backend.forward(torch.from_numpy(x)).numpy()
                            self.assertEqual(actual.shape, expected.shape)
                            np.testing.assert_allclose(
                                actual, expected, atol=0.003, rtol=1e-4
                            )
                            key = ImageShape(height=64, width=96, batch_size=batch_size)
                            compiled = backend._compiled_models[key]
                            if key in cached:
                                self.assertIs(compiled, cached[key])
                            cached[key] = compiled
                            self.assertEqual(len(backend._compiled_models), len(cached))
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
            images = list(
                np.random.default_rng(41).integers(
                    0, 256, size=(2, 64, 64, 3), dtype=np.uint8
                )
            )
            expected = model.predict(source=images, batch=2, imgsz=64, verbose=False)
            max_model = MoloYOLO(artifact, task="detect")
            actual = max_model.predict(source=images, batch=2, imgsz=64, verbose=False)
            self.assertEqual(len(actual), len(images))
            for result, reference in zip(actual, expected):
                assert result.boxes is not None and reference.boxes is not None
                np.testing.assert_allclose(
                    np.asarray(result.boxes.data),
                    np.asarray(reference.boxes.data),
                    atol=0.003,
                    rtol=1e-4,
                )
            assert max_model.predictor is not None
            backend = max_model.predictor.model.backend
            self.assertIn(
                ImageShape(height=64, width=64, batch_size=2), backend._compiled_models
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
                    reference = (
                        DetectionModel(cfg, nc=3, verbose=False)
                        .eval()
                        .fuse(verbose=False)
                    )
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

    def test_detection_head_metadata(self):
        for model_id, branch, end2end in (
            ("yolo11n", "cv2", False),
            ("yolo26n", "one2one_cv2", True),
        ):
            with self.subTest(model_id=model_id):
                artifact = YOLOArtifact(
                    model_id=model_id,
                    class_names=["object"],
                    weights={
                        f"model.23.{branch}.0.2.weight": np.zeros(
                            (4, 16, 1, 1), dtype=np.float32
                        )
                    },
                )
                with (
                    patch(
                        "molo.backends.max.pipelines.architectures.yolo.artifact.load_artifact",
                        return_value=artifact,
                    ),
                    patch("max.engine.InferenceSession"),
                ):
                    backend = MaxBackend(f"{model_id}.max", device=torch.device("cpu"))
                assert backend.config is not None
                self.assertEqual(backend.config.reg_max, 1)
                self.assertEqual(backend.names, {0: "object"})
                self.assertEqual(backend.end2end, end2end)
                self.assertEqual(backend.metadata["end2end"], end2end)
