# Copyright (C) 2026 RidgeRun, LLC (http://www.ridgerun.com)
# SPDX-License-Identifier: AGPL-3.0-only

import unittest
import numpy as np
import torch
from ultralytics.nn.modules.block import DFL as Reference
from max.driver import Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import Graph, TensorType
from molo.backends.max.pipelines.architectures.yolo.layers.dfl import DFL
from tests.helpers import DEVICE, REF


class DFLTest(unittest.TestCase):
    def test_logits(self):
        shape = (2, 64, 11)
        reference = Reference().eval()
        layer = DFL(device=REF)
        weights = {
            key: value.detach().numpy() for key, value in reference.state_dict().items()
        }
        layer.load_state_dict(weights, strict=True)
        with Graph(
            "dfl_test", input_types=[TensorType(DType.float32, shape, REF)]
        ) as graph:
            graph.output(layer(graph.inputs[0].tensor))
        model = InferenceSession(devices=[DEVICE]).load(graph, weights_registry=weights)
        reference = Reference().eval()
        rng = np.random.default_rng(42)
        for label, x in [
            ("random", rng.normal(size=shape).astype(np.float32)),
            ("large", rng.normal(0, 80, size=shape).astype(np.float32)),
            ("equal", np.zeros(shape, np.float32)),
        ]:
            with self.subTest(logits=label):
                actual = model.execute(Buffer.from_numpy(x).to(DEVICE))[0].to_numpy()
                expected = reference(torch.from_numpy(x)).detach().numpy()
                np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=1e-5)
