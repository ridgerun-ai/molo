# Copyright (C) 2026 RidgeRun, LLC (http://www.ridgerun.com)
# SPDX-License-Identifier: AGPL-3.0-only

"""Compare YOLO26 decoding and NMS-free selection with Ultralytics."""

import unittest

import numpy as np
import torch
from ultralytics.nn.modules.head import Detect as Reference
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import Graph, TensorType
from molo.backends.max.pipelines.architectures.yolo.convert import assign_module
from molo.backends.max.pipelines.architectures.yolo26.layers.detect import YOLO26Detect
from tests.helpers import DEVICE, REF, prepare


class DetectTest(unittest.TestCase):
    def test_raw_decoded_and_detections(self):
        for batch, classes in ((1, 5), (2, 2)):
            with self.subTest(batch=batch, classes=classes):
                channels = (8, 16, 32)
                shapes = [
                    (batch, c, h, w) for c, h, w in ((8, 4, 6), (16, 2, 3), (32, 1, 2))
                ]
                reference = prepare(
                    Reference(nc=classes, ch=channels, reg_max=1, end2end=True)
                )
                reference.stride = torch.tensor([8.0, 16.0, 32.0])
                reference.fuse()
                weights = {}
                assign_module(weights, "", reference)
                model = YOLO26Detect(classes, channels, device=REF)
                model.load_state_dict(weights, strict=True)
                self.assertEqual(set(model.raw_state_dict()), set(weights))
                self.assertNotIn("cv2", model.sublayers)
                self.assertNotIn("cv3", model.sublayers)
                with Graph(
                    "yolo26_head_test",
                    input_types=[
                        TensorType(DType.float32, shape, REF) for shape in shapes
                    ],
                ) as graph:
                    raw = model.raw([value.tensor for value in graph.inputs])
                    decoded = model.decode(raw)
                    graph.output(model.postprocess(decoded), decoded, *raw)
                compiled = InferenceSession(devices=[DEVICE]).load(
                    graph, weights_registry=weights
                )
                inputs = [
                    np.random.default_rng(i).normal(size=shape).astype(np.float32)
                    for i, shape in enumerate(shapes)
                ]
                actual = [np.from_dlpack(value) for value in compiled.execute(*inputs)]
                with torch.inference_mode():
                    features = [torch.from_numpy(value) for value in inputs]
                    raw_ref = [
                        torch.cat(
                            (
                                reference.one2one_cv2[i](value),
                                reference.one2one_cv3[i](value),
                            ),
                            dim=1,
                        )
                        for i, value in enumerate(features)
                    ]
                    logits = reference.forward_head(features, **reference.one2one)
                    decoded_ref = reference._inference(logits)
                    detections_ref = reference(features)[0]
                for result, expected in zip(
                    actual,
                    [
                        detections_ref.numpy(),
                        decoded_ref.numpy(),
                        *[value.numpy() for value in raw_ref],
                    ],
                ):
                    np.testing.assert_allclose(result, expected, atol=2e-4, rtol=4e-5)

    def test_postprocess(self):
        for batch, classes, anchors in ((1, 3, 9), (2, 5, 320)):
            with self.subTest(batch=batch, anchors=anchors):
                values = np.random.default_rng(25).random(
                    (batch, 4 + classes, anchors), dtype=np.float32
                )
                model = YOLO26Detect(classes, (8,), strides=(8,), device=REF)
                with Graph(
                    "yolo26_postprocess_test",
                    input_types=[TensorType(DType.float32, values.shape, REF)],
                ) as graph:
                    graph.output(model.postprocess(graph.inputs[0].tensor))
                compiled = InferenceSession(devices=[DEVICE]).load(graph)
                actual = np.from_dlpack(compiled.execute(values)[0])
                reference = Reference(nc=classes, ch=(8,), reg_max=1, end2end=True)
                expected = reference.postprocess(
                    torch.from_numpy(values).permute(0, 2, 1)
                ).numpy()
                np.testing.assert_array_equal(actual, expected)
