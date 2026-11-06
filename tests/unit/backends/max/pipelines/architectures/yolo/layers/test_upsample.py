import unittest
from torch import nn
from molo.backends.max.pipelines.architectures.yolo.layers.upsample import (
    UpsampleNearest2x,
)
from tests.unit.helpers import compare


class UpsampleTest(unittest.TestCase):
    def test_rectangular_and_batch(self):
        for shape in [(1, 3, 2, 5), (2, 4, 1, 1)]:
            with self.subTest(shape=shape):
                compare(
                    UpsampleNearest2x(),
                    nn.Upsample(scale_factor=2, mode="nearest"),
                    shape,
                    prefix="",
                )
