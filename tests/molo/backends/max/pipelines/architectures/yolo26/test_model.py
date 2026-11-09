# Copyright (C) 2026 RidgeRun, LLC (http://www.ridgerun.com)
# SPDX-License-Identifier: AGPL-3.0-only

"""Verify YOLO26 architecture, conversion, and graph inference."""

from copy import deepcopy
from typing import cast
import unittest
from unittest.mock import patch

import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
from max.engine import InferenceSession

from molo.backends.max.pipelines.architectures.yolo.config import ImageShape, YOLOConfig
from molo.backends.max.pipelines.architectures.yolo26.model import (
    YOLO26,
    YOLO26_SIZES,
    build_graph,
    convert_model,
    identify_model,
)
from molo.backends.max.pipelines.architectures.yolo26.layers.detect import YOLO26Detect
from tests.helpers import DEVICE, module_paths


class ModelTest(unittest.TestCase):
    def test_sizes_and_weight_structure(self):
        for size, scale in YOLO26_SIZES.items():
            with self.subTest(size=size):
                reference = (
                    cast(DetectionModel, YOLO(f"yolo26{size}.yaml").model)
                    .eval()
                    .fuse(verbose=False)
                )
                config = identify_model(reference)
                self.assertEqual(config.size, size)
                self.assertEqual(config.reg_max, 1)
                self.assertEqual(
                    (scale.depth, scale.width, scale.max_channels),
                    tuple(reference.yaml["scales"][size]),
                )
                original_keys = set(reference.state_dict())
                with patch(
                    "builtins.open",
                    side_effect=AssertionError("Conversion must not write files"),
                ):
                    artifact = convert_model(reference)
                self.assertEqual(artifact.model_id, f"yolo26{size}")
                self.assertEqual(
                    artifact.class_names, [reference.names[i] for i in range(config.nc)]
                )
                self.assertEqual(set(reference.state_dict()), original_keys)
                model = YOLO26(config)
                self.assertIs(model.config, config)
                expected_paths = {
                    name
                    for name, _ in reference.named_modules(remove_duplicate=False)
                    if name
                }
                self.assertEqual(module_paths(model), expected_paths)
                expected = reference.state_dict()
                self.assertEqual(set(model.raw_state_dict()), set(expected))
                self.assertEqual(set(artifact.weights), set(expected))
                model.load_state_dict(artifact.weights, strict=True)
                for name, value in model.state_dict(auto_initialize=False).items():
                    np.testing.assert_array_equal(
                        value, expected[name].detach().numpy()
                    )
                # Check graph construction and batch propagation for every size.
                graph = build_graph(
                    config, artifact, ImageShape(height=64, width=96, batch_size=2)
                )
                self.assertEqual(list(graph.inputs[0].tensor.shape), [2, 3, 64, 96])

    def test_custom_classes(self):
        yaml = deepcopy(cast(DetectionModel, YOLO("yolo26n.yaml").model).yaml)
        reference = DetectionModel(yaml, nc=5, verbose=False).eval().fuse(verbose=False)
        config = identify_model(reference)
        self.assertEqual(config.nc, 5)
        model = YOLO26(config)
        model.load_state_dict(convert_model(reference).weights, strict=True)
        self.assertEqual(cast(YOLO26Detect, model.model[23]).num_classes, 5)

    def test_invalid_size(self):
        with self.assertRaisesRegex(ValueError, "Unsupported YOLO26 size"):
            YOLO26(YOLOConfig(size="bad", reg_max=1))  # ty: ignore[invalid-argument-type]
        reference = cast(DetectionModel, YOLO("yolo26n.yaml").model)
        reference.yaml["scale"] = "bad"
        with self.assertRaisesRegex(ValueError, "Unsupported yolo26 size"):
            identify_model(reference)

    def test_rejects_non_end_to_end_model(self):
        reference = (
            cast(DetectionModel, YOLO("yolo26n.yaml").model).eval().fuse(verbose=False)
        )
        reference.model[-1].end2end = False
        with (
            patch(
                "molo.backends.max.pipelines.architectures.yolo26.model.assign_module"
            ) as extract_weights,
            self.assertRaisesRegex(ValueError, "requires end2end=True"),
        ):
            convert_model(reference)
        extract_weights.assert_not_called()

    def test_rejects_non_default_selection(self):
        for option, value, message in (
            ("max_det", 10, "requires max_det=300"),
            ("max_det", 400, "requires max_det=300"),
            ("agnostic_nms", True, "requires agnostic_nms=False"),
        ):
            with self.subTest(option=option, value=value):
                reference = (
                    cast(DetectionModel, YOLO("yolo26n.yaml").model)
                    .eval()
                    .fuse(verbose=False)
                )
                setattr(reference.model[-1], option, value)
                with (
                    patch(
                        "molo.backends.max.pipelines.architectures.yolo26.model.assign_module"
                    ) as extract_weights,
                    self.assertRaisesRegex(ValueError, message),
                ):
                    convert_model(reference)
                extract_weights.assert_not_called()

    def test_rejects_unprepared_model(self):
        for state in ("unfused", "training"):
            with self.subTest(state=state):
                reference = cast(DetectionModel, YOLO("yolo26n.yaml").model).eval()
                if state == "training":
                    reference.fuse(verbose=False).train()
                with (
                    patch(
                        "molo.backends.max.pipelines.architectures.yolo26.model.assign_module"
                    ) as extract_weights,
                    self.assertRaisesRegex(
                        ValueError, "requires a fused evaluation model"
                    ),
                ):
                    convert_model(reference)
                extract_weights.assert_not_called()

    def test_prepared_conversion_does_not_fuse(self):
        reference = (
            cast(DetectionModel, YOLO("yolo26n.yaml").model).eval().fuse(verbose=False)
        )
        with (
            patch.object(
                reference,
                "fuse",
                side_effect=AssertionError("Model must already be fused"),
            ),
            patch.object(
                reference.model[-1],
                "fuse",
                side_effect=AssertionError("Head must already be fused"),
            ),
        ):
            artifact = convert_model(reference)
        self.assertEqual(set(artifact.weights), set(reference.state_dict()))

    def test_strict_loading_and_precision(self):
        reference = (
            cast(DetectionModel, YOLO("yolo26n.yaml").model).eval().fuse(verbose=False)
        )
        config = identify_model(reference)
        artifact = convert_model(reference)
        key = "model.23.one2one_cv2.0.2.weight"
        for corruption in ("missing", "unexpected", "shape", "dtype"):
            with self.subTest(corruption=corruption):
                weights = dict(artifact.weights)
                if corruption == "missing":
                    del weights[key]
                elif corruption == "unexpected":
                    weights["unexpected.weight"] = np.zeros(1, dtype=np.float32)
                elif corruption == "shape":
                    weights[key] = np.zeros(1, dtype=np.float32)
                else:
                    weights[key] = weights[key].astype(np.float64)
                with self.assertRaises(ValueError):
                    YOLO26(config).load_state_dict(weights, strict=True)
        with self.assertRaisesRegex(ValueError, "float32"):
            convert_model(reference.half())

    def test_checkpoint_predictions(self):
        reference = (
            cast(DetectionModel, YOLO("yolo26n.pt").model).eval().fuse(verbose=False)
        )
        artifact = convert_model(reference)
        config = identify_model(reference)
        for batch in (1, 2):
            with self.subTest(batch=batch):
                shape = ImageShape(height=64, width=96, batch_size=batch)
                values = np.random.default_rng(42).random(
                    (batch, 3, 64, 96), dtype=np.float32
                )
                graph = build_graph(config, artifact, shape)
                compiled = InferenceSession(devices=[DEVICE]).load(
                    graph, weights_registry=artifact.weights
                )
                actual = np.from_dlpack(compiled.execute(values)[0])
                with torch.inference_mode():
                    expected = reference(torch.from_numpy(values))[0].numpy()
                self.assertEqual(actual.shape, expected.shape)
                np.testing.assert_allclose(actual, expected, atol=0.003, rtol=1e-4)
