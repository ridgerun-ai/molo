"""Tests for shared YOLO checkpoint conversion."""

import unittest
from types import SimpleNamespace
from typing import cast
from pathlib import Path

from molo.backends.max.pipelines.architectures.yolo.artifact import YOLOArtifact
from unittest.mock import Mock, patch

import numpy as np
import torch

from molo.backends.max.pipelines.architectures.yolo.convert import (
    assign_module,
    convert_model,
)


class ConvertTest(unittest.TestCase):
    def test_assign_module(self):
        module = torch.nn.BatchNorm2d(3)
        weights = {}
        assign_module(weights, "layer", module)
        self.assertNotIn("layer.num_batches_tracked", weights)
        assert module.running_mean is not None
        np.testing.assert_array_equal(
            weights["layer.running_mean"], module.running_mean.numpy()
        )
        np.testing.assert_array_equal(
            weights["layer.weight"], module.weight.detach().numpy()
        )

    def test_fused_convolution_weights(self):
        from ultralytics.nn.modules.conv import Conv
        from tests.helpers import prepare

        module = prepare(Conv(3, 8, 3))
        x = torch.randn(1, 3, 5, 7)
        weights = {}
        assign_module(weights, "layer", module)
        self.assertEqual(set(weights), {"layer.conv.weight", "layer.conv.bias"})
        actual = torch.nn.functional.silu(
            torch.nn.functional.conv2d(
                x,
                torch.from_numpy(weights["layer.conv.weight"]),
                torch.from_numpy(weights["layer.conv.bias"]),
                padding=1,
            )
        )
        with torch.no_grad():
            torch.testing.assert_close(actual, module(x))
        self.assertTrue(hasattr(module, "bn"))

    def test_rejects_fp16(self):
        from ultralytics import YOLO
        from ultralytics.nn.tasks import DetectionModel

        for source in ("yolo11n.yaml", "yolov8n.yaml"):
            with self.subTest(source=source):
                model = cast(DetectionModel, YOLO(source).model).half()
                with (
                    patch(
                        "molo.backends.max.pipelines.architectures.yolo.convert.save_artifact"
                    ) as save,
                    self.assertRaisesRegex(ValueError, "requires float32 weights"),
                ):
                    convert_model(model, "unsupported.max", model_id=Path(source).stem)
                save.assert_not_called()

    def test_dispatch(self):
        for prefix in ("yolo11", "yolov8", "another_architecture"):
            model_id = f"{prefix}n"
            with self.subTest(model_id=model_id):
                model = Mock(
                    model="layers",
                    names={0: "object"},
                    yaml={"yaml_file": f"{model_id}.yaml"},
                    args={},
                )
                artifact = YOLOArtifact(model_id, ["object"], {})
                architecture = SimpleNamespace(
                    convert_model=Mock(return_value=artifact)
                )
                with (
                    patch(
                        "molo.backends.max.pipelines.architectures.get_architecture",
                        return_value=architecture,
                    ) as factory,
                    patch(
                        "molo.backends.max.pipelines.architectures.yolo.convert.save_artifact"
                    ) as save,
                ):
                    convert_model(model, "destination.max", model_id=model_id)
                factory.assert_called_once_with(model_id)
                architecture.convert_model.assert_called_once_with(model)
                save.assert_called_once_with("destination.max", artifact)
