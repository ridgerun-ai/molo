# Copyright (C) 2026 RidgeRun, LLC (http://www.ridgerun.com)
# SPDX-License-Identifier: AGPL-3.0-only

"""Compare BatchNorm with the BatchNorm used by Ultralytics."""

import unittest
import torch
from molo.backends.max.nn.norm.batch_norm import BatchNorm
import numpy as np
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import Graph, TensorType
from tests.helpers import DEVICE, REF, prepare


class BatchNormTest(unittest.TestCase):
    def test_inference_ranks_and_affine(self):
        for shape, affine in [
            ((2, 4), True),
            ((1, 4, 7), False),
            ((2, 4, 3, 5), True),
            ((1, 4, 2, 3, 4), False),
        ]:
            with self.subTest(shape=shape, affine=affine):
                cls = {
                    2: torch.nn.BatchNorm1d,
                    3: torch.nn.BatchNorm1d,
                    4: torch.nn.BatchNorm2d,
                    5: torch.nn.BatchNorm3d,
                }[len(shape)]
                reference = prepare(cls(4, affine=affine))
                layer = BatchNorm(4, eps=0.001, affine=affine, device=REF, name="layer")
                state = reference.state_dict()
                names = ["running_mean", "running_var"]
                if affine:
                    names += ["weight", "bias"]
                weights = {
                    f"layer.{name}": state[name].detach().numpy() for name in names
                }
                layer.load_state_dict(weights, strict=True)
                with Graph(
                    "batch_norm_test",
                    input_types=[TensorType(DType.float32, shape, REF)],
                ) as graph:
                    graph.output(layer(graph.inputs[0].tensor))
                compiled = InferenceSession(devices=[DEVICE]).load(
                    graph, weights_registry=weights
                )
                values = np.random.default_rng(17).normal(size=shape).astype(np.float32)
                actual = np.from_dlpack(compiled.execute(values)[0])
                with torch.inference_mode():
                    expected = reference(torch.from_numpy(values)).numpy()
                np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=3e-5)

    def test_invalid_parameters(self):
        with self.assertRaises(ValueError):
            BatchNorm(0, device=REF)
        with self.assertRaises(ValueError):
            BatchNorm(4, eps=0, device=REF)
