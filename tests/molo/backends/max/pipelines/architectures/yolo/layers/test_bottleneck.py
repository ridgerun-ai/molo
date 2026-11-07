import unittest
from ultralytics.nn.modules.block import Bottleneck as Reference
from molo.backends.max.pipelines.architectures.yolo.layers.bottleneck import (
    Bottleneck,
)
from tests.helpers import REF, compare


class BottleneckTest(unittest.TestCase):
    def test_configurations(self):
        for ci, co, shortcut, groups, kernels, expansion in [
            (8, 8, True, 1, (3, 3), 0.5),
            (8, 8, False, 1, (3, 3), 0.5),
            (8, 12, True, 1, (1, 3), 0.5),
            (8, 8, True, 2, (3, 3), 1.0),
        ]:
            with self.subTest(ci=ci, co=co, shortcut=shortcut, groups=groups):
                compare(
                    Bottleneck(
                        ci,
                        co,
                        shortcut,
                        groups,
                        kernels,
                        expansion,
                        device=REF,
                    ),
                    Reference(
                        ci, co, shortcut=shortcut, g=groups, k=kernels, e=expansion
                    ),
                    (1, ci, 5, 7),
                )
