# Copyright (C) 2026 RidgeRun, LLC (http://www.ridgerun.com)
# SPDX-License-Identifier: AGPL-3.0-only

import unittest
from molo.backends.max.pipelines.architectures.yolo.convert import assign_module
import numpy as np
import torch
from ultralytics.nn.modules.head import Detect as Reference
from max.driver import Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import Graph, TensorType
from molo.backends.max.pipelines.architectures.yolov8.layers.detect import YOLOv8Detect
from tests.helpers import DEVICE, REF, prepare


class YOLOv8Reference(Reference):
    legacy = True


class DetectTest(unittest.TestCase):
    def test_raw_and_decoded(self):
        self.check_head(YOLOv8Detect, YOLOv8Reference)

    def test_reg_max(self):
        for reg_max in (1, 4, 8, 20):
            with self.subTest(reg_max=reg_max):
                self.check_head(YOLOv8Detect, YOLOv8Reference, reg_max=reg_max)

    def check_head(self, model_type, reference_type, reg_max=16):
        for classes, channels, shapes in [
            (5, (8, 16, 32), ((1, 8, 4, 6), (1, 16, 2, 3), (1, 32, 1, 2))),
            (2, (8, 16, 32), ((1, 8, 2, 2), (1, 16, 1, 1), (1, 32, 1, 1))),
        ]:
            with self.subTest(classes=classes, shapes=shapes):
                reference = prepare(
                    reference_type(nc=classes, ch=channels, reg_max=reg_max)
                )
                reference.stride = torch.tensor([8.0, 16.0, 32.0])
                weights = {}
                assign_module(weights, "", reference)
                model = model_type(classes, channels, device=REF, reg_max=reg_max)
                model.load_state_dict(weights, strict=True)
                self.assertEqual(
                    {weight.name for weight in model.raw_state_dict().values()},
                    set(weights),
                )
                with Graph(
                    "detect_test",
                    input_types=[TensorType(DType.float32, s, REF) for s in shapes],
                ) as graph:
                    raw = model.raw([x.tensor for x in graph.inputs])
                    graph.output(model.decode(raw), *raw)
                compiled = InferenceSession(devices=[DEVICE]).load(
                    graph, weights_registry=weights
                )
                inputs = [
                    np.random.default_rng(i).normal(size=s).astype(np.float32)
                    for i, s in enumerate(shapes)
                ]
                actual = [
                    v.to_numpy()
                    for v in compiled.execute(
                        *[Buffer.from_numpy(x).to(DEVICE) for x in inputs]
                    )
                ]
                with torch.inference_mode():
                    features = [torch.from_numpy(x) for x in inputs]
                    raw_ref = [
                        torch.cat((reference.cv2[i](x), reference.cv3[i](x)), dim=1)
                        for i, x in enumerate(features)
                    ]
                    decoded, _ = reference(features)
                for a, b in zip(
                    actual, [decoded.numpy(), *[x.numpy() for x in raw_ref]]
                ):
                    np.testing.assert_allclose(a, b, atol=0.0002, rtol=4e-5)
