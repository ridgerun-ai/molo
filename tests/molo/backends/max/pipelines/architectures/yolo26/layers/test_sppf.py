"""Compare YOLO26 residual pooling with Ultralytics."""

import unittest

from ultralytics.nn.modules.block import SPPF as Reference
from molo.backends.max.pipelines.architectures.yolo26.layers.sppf import SPPF
from tests.helpers import REF, compare


class SPPFTest(unittest.TestCase):
    def test_configurations(self):
        for channels, kernel, shape in [
            (8, 5, (1, 8, 5, 7)),
            (12, 3, (2, 12, 3, 4)),
            (8, 5, (1, 8, 1, 1)),
        ]:
            with self.subTest(channels=channels, kernel=kernel, shape=shape):
                compare(
                    SPPF(channels, kernel, device=REF),
                    Reference(channels, channels, k=kernel, n=3, shortcut=True),
                    shape,
                )
