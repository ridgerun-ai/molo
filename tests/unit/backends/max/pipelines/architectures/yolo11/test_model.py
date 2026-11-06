"""Compare YOLO11 size scaling with Ultralytics."""

import unittest
from unittest.mock import Mock, patch
from typing import cast
from molo.backends.max.pipelines.architectures.yolo.config import YOLOConfig
from pathlib import Path

import numpy as np
import torch

from max.graph import DeviceRef
from ultralytics import YOLO
from molo.backends.max.pipelines.architectures.yolo11.model import (
    convert_model as convert_weights,
    identify_model,
    YOLO11_SIZES,
)

from molo.backends.max.pipelines.architectures.yolo.convert import convert_model
from molo.models.yolo.backend import MaxBackend
from ultralytics.nn.tasks import DetectionModel

from molo.backends.max.pipelines.architectures.yolo11 import YOLO11


class ModelSizesTest(unittest.TestCase):
    def test_architecture_scaling(self):
        for size in ("n", "s", "m", "l", "x"):
            with self.subTest(size=size):
                model = YOLO11(YOLOConfig(size=size, device=DeviceRef.CPU()))
                reference = YOLO(f"yolo11{size}.yaml").model.model  # ty: ignore[unresolved-attribute]
                for i in (0, 1, 3, 5, 7, 17, 20):
                    actual = getattr(model, f"layer{i}").conv
                    expected = reference[i].conv
                    self.assertEqual(
                        (actual.in_channels, actual.out_channels),
                        (expected.in_channels, expected.out_channels),
                    )
                for i in (2, 4, 6, 8, 13, 16, 19, 22):
                    actual = getattr(model, f"layer{i}")
                    expected = reference[i]
                    self.assertEqual(len(actual.m), len(expected.m))
                    self.assertEqual(
                        actual.cv1.conv.out_channels, expected.cv1.conv.out_channels
                    )
                    self.assertEqual(
                        type(actual.m[0]).__name__, type(expected.m[0]).__name__
                    )
                self.assertEqual(len(model.layer10.m), len(reference[10].m))

    def test_custom_class_count_scaling(self):
        for size in ("n", "m", "x"):
            with self.subTest(size=size):
                yaml = YOLO(f"yolo11{size}.yaml").model.yaml  # ty: ignore[unresolved-attribute]
                reference = DetectionModel(yaml, ch=3, nc=64, verbose=False)
                model = YOLO11(YOLOConfig(size=size, nc=64))
                for i in (0, 1, 3, 5, 7, 17, 20):
                    actual = getattr(model, f"layer{i}").conv
                    expected = reference.model[i].conv
                    self.assertEqual(
                        (actual.in_channels, actual.out_channels),
                        (expected.in_channels, expected.out_channels),
                    )

    def test_invalid_size(self):
        with self.assertRaisesRegex(ValueError, "Unsupported YOLO11 size"):
            YOLO11(YOLOConfig(size="bad"))  # ty: ignore[invalid-argument-type]

    def test_checkpoint_predictions(self):
        x = np.random.default_rng(24).random((1, 3, 64, 64), dtype=np.float32)
        for size in ("n", "s", "m", "l", "x"):
            with self.subTest(size=size):
                source = Path(f"yolo11{size}.pt")
                artifact = source.with_suffix(".max")
                if artifact.exists():
                    raise FileExistsError(f"Refusing to overwrite {artifact}")
                reference = YOLO(str(source)).model.eval()  # ty: ignore[unresolved-attribute]
                try:
                    convert_model(reference, artifact, model_id=f"yolo11{size}")
                    backend = MaxBackend(str(artifact), device=torch.device("cpu"))
                    assert backend.config is not None
                    self.assertEqual(backend.config.size, size)
                    with torch.inference_mode():
                        expected = reference(torch.from_numpy(x))[0].numpy()
                        actual = backend.forward(torch.from_numpy(x)).numpy()
                    np.testing.assert_allclose(actual, expected, atol=0.003, rtol=1e-4)
                finally:
                    artifact.unlink(missing_ok=True)

    def test_weight_names(self):
        layers = [torch.nn.Linear(1, 1) for _ in range(24)]
        setattr(layers[-1], "reg_max", 16)
        model = Mock(
            model=layers, names={1: "second", 0: "first"}, yaml={"scale": "n", "nc": 2}
        )
        with patch(
            "builtins.open", side_effect=AssertionError("Conversion must not write")
        ):
            artifact = convert_weights(model)
        self.assertEqual(artifact.model_id, "yolo11n")
        self.assertEqual(artifact.class_names, ["first", "second"])
        self.assertIn("layer0.weight", artifact.weights)
        self.assertIn("detect.weight", artifact.weights)

    def test_identify_model(self):
        for size in ("n", "s", "m", "l", "x"):
            for extension in ("yaml", "pt"):
                with self.subTest(size=size, extension=extension):
                    reference = cast(
                        DetectionModel, YOLO(f"yolo11{size}.{extension}").model
                    )
                    reference.yaml.pop("yaml_file", None)
                    config = identify_model(reference)
                    self.assertIsInstance(config, YOLOConfig)
                    self.assertEqual(config.size, size)
                    self.assertEqual(config.nc, reference.yaml["nc"])
                    self.assertEqual(config.reg_max, reference.model[-1].reg_max)
                    model = YOLO11(config)
                    self.assertIs(model.config, config)
                    self.assertEqual(model.detect.num_classes, config.nc)

    def test_identify_custom_classes(self):
        reference = cast(DetectionModel, YOLO("yolo11n.yaml").model)
        reference = DetectionModel(reference.yaml, nc=5, verbose=False)
        config = identify_model(reference)
        self.assertEqual(config.nc, 5)
        self.assertEqual(YOLO11(config).detect.num_classes, 5)

    def test_sizes(self):
        for size, scale in YOLO11_SIZES.items():
            with self.subTest(size=size):
                reference = YOLO(f"yolo11{size}.yaml").model.yaml["scales"][size]  # ty: ignore[unresolved-attribute]
                self.assertEqual(
                    (scale.depth, scale.width, scale.max_channels), tuple(reference)
                )
