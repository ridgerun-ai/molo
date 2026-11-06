"""Tests for shared YOLO checkpoint conversion."""

import unittest
from types import SimpleNamespace

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

    def test_dispatch(self):
        for prefix in ("yolo11", "another_architecture"):
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
