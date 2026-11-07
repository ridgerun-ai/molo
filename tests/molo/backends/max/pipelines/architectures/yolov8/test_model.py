"""Compare YOLOv8 model scaling with Ultralytics."""

import unittest
from molo.backends.max.pipelines.architectures.yolo.layers.bottleneck import Bottleneck
from molo.backends.max.pipelines.architectures.yolo.layers.conv import Conv
from molo.backends.max.pipelines.architectures.yolov8.layers.c2f import C2f
from molo.backends.max.pipelines.architectures.yolov8.layers.detect import YOLOv8Detect
from tests.helpers import module_paths
from unittest.mock import Mock, patch
from typing import cast
from ultralytics.nn.tasks import DetectionModel
from molo.backends.max.pipelines.architectures.yolo.config import YOLOConfig
from pathlib import Path

import numpy as np
import torch

from ultralytics import YOLO
from molo.backends.max.pipelines.architectures.yolov8.model import (
    convert_model as convert_weights,
    identify_model,
    YOLOV8_SIZES,
)

from molo.backends.max.pipelines.architectures.yolo.convert import convert_model
from molo.models.yolo.backend import MaxBackend

from molo.backends.max.pipelines.architectures.yolov8 import YOLOv8


class ModelSizesTest(unittest.TestCase):
    def test_architecture_scaling(self):
        for size in ("n", "s", "m", "l", "x"):
            with self.subTest(size=size):
                model = YOLOv8(YOLOConfig(size=size))
                reference = YOLO(f"yolov8{size}.yaml").model.model  # ty: ignore[unresolved-attribute]
                for index in (0, 1, 3, 5, 7, 16, 19):
                    actual = cast(Conv, model.model[index]).conv
                    expected = reference[index].conv
                    self.assertEqual(
                        (actual.in_channels, actual.out_channels),
                        (expected.in_channels, expected.out_channels),
                    )
                for index in (2, 4, 6, 8, 12, 15, 18, 21):
                    actual = cast(C2f, model.model[index])
                    expected = reference[index]
                    self.assertEqual(len(actual.m), len(expected.m))
                    self.assertEqual(
                        actual.cv1.conv.out_channels, expected.cv1.conv.out_channels
                    )
                    self.assertEqual(
                        cast(Bottleneck, actual.m[0]).cv1.conv.out_channels,
                        expected.m[0].cv1.conv.out_channels,
                    )

    def test_checkpoint_predictions(self):
        x = np.random.default_rng(24).random((1, 3, 64, 64), dtype=np.float32)
        for size in ("n", "s", "m", "l", "x"):
            with self.subTest(size=size):
                source = Path(f"yolov8{size}.pt")
                artifact = source.with_suffix(".max")
                if artifact.exists():
                    raise FileExistsError(f"Refusing to overwrite {artifact}")
                reference = YOLO(str(source)).model.eval()  # ty: ignore[unresolved-attribute]
                try:
                    convert_model(reference, artifact, model_id=f"yolov8{size}")
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
        layers = [torch.nn.Linear(1, 1) for _ in range(23)]
        setattr(layers[-1], "reg_max", 16)
        model = Mock(
            model=torch.nn.Sequential(*layers),
            names={1: "second", 0: "first"},
            yaml={"scale": "n", "nc": 2},
        )
        with patch(
            "builtins.open", side_effect=AssertionError("Conversion must not write")
        ):
            artifact = convert_weights(model)
        self.assertEqual(artifact.model_id, "yolov8n")
        self.assertEqual(artifact.class_names, ["first", "second"])
        self.assertIn("model.0.weight", artifact.weights)
        self.assertIn("model.22.weight", artifact.weights)

    def test_identify_model(self):
        for size in ("n", "s", "m", "l", "x"):
            for extension in ("yaml", "pt"):
                with self.subTest(size=size, extension=extension):
                    reference = cast(
                        DetectionModel, YOLO(f"yolov8{size}.{extension}").model
                    )
                    reference.yaml.pop("yaml_file", None)
                    config = identify_model(reference)
                    self.assertIsInstance(config, YOLOConfig)
                    self.assertEqual(config.size, size)
                    self.assertEqual(config.nc, reference.yaml["nc"])
                    self.assertEqual(config.reg_max, reference.model[-1].reg_max)
                    model = YOLOv8(config)
                    self.assertIs(model.config, config)
                    reference.fuse(verbose=False)
                    expected_paths = {
                        path
                        for path, _ in reference.named_modules(remove_duplicate=False)
                        if path
                    }
                    self.assertEqual(module_paths(model), expected_paths)
                    artifact = convert_weights(reference)
                    expected_weights = {
                        name: value for name, value in reference.state_dict().items()
                    }
                    self.assertEqual(set(model.raw_state_dict()), set(expected_weights))
                    self.assertEqual(set(artifact.weights), set(expected_weights))
                    model.load_state_dict(artifact.weights, strict=True)
                    for name, weight in model.raw_state_dict().items():
                        self.assertEqual(weight.name, name)
                    self.assertEqual(
                        set(model.state_dict(auto_initialize=False)),
                        set(expected_weights),
                    )
                    for name, value in model.state_dict(auto_initialize=False).items():
                        np.testing.assert_array_equal(
                            value, expected_weights[name].detach().numpy()
                        )

                    self.assertEqual(
                        cast(YOLOv8Detect, model.model[22]).num_classes, config.nc
                    )

    def test_identify_custom_classes(self):
        reference = cast(DetectionModel, YOLO("yolov8n.yaml").model)
        reference = DetectionModel(reference.yaml, nc=5, verbose=False)
        config = identify_model(reference)
        self.assertEqual(config.nc, 5)
        self.assertEqual(cast(YOLOv8Detect, YOLOv8(config).model[22]).num_classes, 5)

    def test_sizes(self):
        for size, scale in YOLOV8_SIZES.items():
            with self.subTest(size=size):
                reference = YOLO(f"yolov8{size}.yaml").model.yaml["scales"][size]  # ty: ignore[unresolved-attribute]
                self.assertEqual(
                    (scale.depth, scale.width, scale.max_channels), tuple(reference)
                )

    def test_strict_weight_loading(self):
        reference = cast(DetectionModel, YOLO("yolov8n.pt").model)
        artifact = convert_weights(reference)
        for corruption in ("missing", "unexpected", "shape", "dtype"):
            with self.subTest(corruption=corruption):
                weights = dict(artifact.weights)
                if corruption == "missing":
                    del weights["model.0.conv.weight"]
                elif corruption == "unexpected":
                    weights["unexpected.weight"] = np.zeros(1, dtype=np.float32)
                elif corruption == "shape":
                    weights["model.0.conv.weight"] = np.zeros(1, dtype=np.float32)
                else:
                    weights["model.0.conv.weight"] = weights[
                        "model.0.conv.weight"
                    ].astype(np.float64)
                model = YOLOv8(identify_model(reference))
                with self.assertRaises(ValueError):
                    model.load_state_dict(weights, strict=True)
