"""Compare BatchNorm with the BatchNorm used by Ultralytics."""

import unittest
import torch
from molo.backends.max.nn.norm.batch_norm import BatchNorm
from tests.helpers import REF, compare


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
                compare(
                    BatchNorm(4, eps=0.001, affine=affine, device=REF, name="layer"),
                    cls(4, affine=affine),
                    shape,
                )

    def test_invalid_parameters(self):
        with self.assertRaises(ValueError):
            BatchNorm(0, device=REF)
        with self.assertRaises(ValueError):
            BatchNorm(4, eps=0, device=REF)
