# Copyright (C) 2026 RidgeRun, LLC (http://www.ridgerun.com)
# SPDX-License-Identifier: AGPL-3.0-only

"""Tests for shared YOLO checkpoint conversion."""

import unittest
from types import SimpleNamespace
from typing import cast
from pathlib import Path

from molo.backends.max.pipelines.architectures.yolo.artifact import YOLOArtifact
from unittest.mock import Mock, patch

import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel

from molo.backends.max.pipelines.architectures.yolo.convert import (
    assign_module,
    convert_model,
)


class ConvertTest(unittest.TestCase):
    def test_assign_module(self):
        module = torch.nn.Conv2d(3, 8, 3).eval()
        weights = {}
        assign_module(weights, "layer", module)
        self.assertEqual(set(weights), {"layer.weight", "layer.bias"})
        for name, value in module.state_dict().items():
            np.testing.assert_array_equal(
                weights[f"layer.{name}"], value.detach().numpy()
            )

    def test_rejects_unfused_module(self):
        module = torch.nn.BatchNorm2d(3).eval()
        weights = {}
        with self.assertRaisesRegex(ValueError, "requires fused weights"):
            assign_module(weights, "layer", module)
        self.assertEqual(weights, {})

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
        self.assertFalse(hasattr(module, "bn"))

    def test_rejects_fp16(self):
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

    def test_prepared_model_conversion(self):
        for family in ("yolo11", "yolov8", "yolo26"):
            with self.subTest(family=family):
                reference = (
                    cast(DetectionModel, YOLO(f"{family}n.yaml").model)
                    .eval()
                    .fuse(verbose=False)
                )
                with patch(
                    "molo.backends.max.pipelines.architectures.yolo.convert.save_artifact"
                ) as save:
                    convert_model(reference, "prepared.max", model_id=f"{family}n")
                artifact = save.call_args.args[1]
                self.assertEqual(artifact.model_id, f"{family}n")
                self.assertEqual(set(artifact.weights), set(reference.state_dict()))
                for name, value in reference.state_dict().items():
                    np.testing.assert_array_equal(artifact.weights[name], value.numpy())

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
