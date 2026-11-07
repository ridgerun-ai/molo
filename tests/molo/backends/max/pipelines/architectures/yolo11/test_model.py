"""Compare YOLO11 size scaling with Ultralytics."""

import unittest
from molo.backends.max.pipelines.architectures.yolo11.layers.c2psa import C2PSA
from molo.backends.max.pipelines.architectures.yolo.layers.conv import Conv
from molo.backends.max.pipelines.architectures.yolo11.layers.c3k2 import C3k2
from molo.backends.max.pipelines.architectures.yolo11.layers.detect import YOLO11Detect
from tests.helpers import module_paths
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
                    actual = cast(Conv, model.model[i]).conv
                    expected = reference[i].conv
                    self.assertEqual(
                        (actual.in_channels, actual.out_channels),
                        (expected.in_channels, expected.out_channels),
                    )
                for i in (2, 4, 6, 8, 13, 16, 19, 22):
                    actual = cast(C3k2, model.model[i])
                    expected = reference[i]
                    self.assertEqual(len(actual.m), len(expected.m))
                    self.assertEqual(
                        actual.cv1.conv.out_channels, expected.cv1.conv.out_channels
                    )
                    self.assertEqual(
                        type(actual.m[0]).__name__, type(expected.m[0]).__name__
                    )
                self.assertEqual(
                    len(cast(C2PSA, model.model[10]).m), len(reference[10].m)
                )

    def test_custom_class_count_scaling(self):
        for size in ("n", "m", "x"):
            with self.subTest(size=size):
                yaml = YOLO(f"yolo11{size}.yaml").model.yaml  # ty: ignore[unresolved-attribute]
                reference = DetectionModel(yaml, ch=3, nc=64, verbose=False)
                model = YOLO11(YOLOConfig(size=size, nc=64))
                for i in (0, 1, 3, 5, 7, 17, 20):
                    actual = cast(Conv, model.model[i]).conv
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
            model=torch.nn.Sequential(*layers),
            names={1: "second", 0: "first"},
            yaml={"scale": "n", "nc": 2},
        )
        with patch(
            "builtins.open", side_effect=AssertionError("Conversion must not write")
        ):
            artifact = convert_weights(model)
        self.assertEqual(artifact.model_id, "yolo11n")
        self.assertEqual(artifact.class_names, ["first", "second"])
        self.assertIn("model.0.weight", artifact.weights)
        self.assertIn("model.23.weight", artifact.weights)

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
                        cast(YOLO11Detect, model.model[23]).num_classes, config.nc
                    )

    def test_identify_custom_classes(self):
        reference = cast(DetectionModel, YOLO("yolo11n.yaml").model)
        reference = DetectionModel(reference.yaml, nc=5, verbose=False)
        config = identify_model(reference)
        self.assertEqual(config.nc, 5)
        self.assertEqual(cast(YOLO11Detect, YOLO11(config).model[23]).num_classes, 5)

    def test_sizes(self):
        for size, scale in YOLO11_SIZES.items():
            with self.subTest(size=size):
                reference = YOLO(f"yolo11{size}.yaml").model.yaml["scales"][size]  # ty: ignore[unresolved-attribute]
                self.assertEqual(
                    (scale.depth, scale.width, scale.max_channels), tuple(reference)
                )

    def test_strict_weight_loading(self):
        reference = cast(DetectionModel, YOLO("yolo11n.pt").model)
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
                model = YOLO11(identify_model(reference))
                with self.assertRaises(ValueError):
                    model.load_state_dict(weights, strict=True)
