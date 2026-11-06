import unittest
from ultralytics.nn.modules.conv import Conv as Reference
from molo.backends.max.pipelines.architectures.yolo.layers.conv import Conv
from tests.helpers import REF, compare


class ConvTest(unittest.TestCase):
    def test_configurations(self):
        for ci, co, k, stride, groups, activation in [
            (3, 8, 3, 2, 1, True),
            (8, 8, 1, 1, 1, False),
            (4, 4, 3, 1, 4, True),
        ]:
            with self.subTest(k=k, stride=stride, groups=groups, activation=activation):
                compare(
                    Conv(
                        ci,
                        co,
                        k,
                        stride,
                        groups=groups,
                        activation=activation,
                        device=REF,
                        name="layer",
                    ),
                    Reference(ci, co, k, stride, g=groups, act=activation),
                    (1, ci, 5, 7),
                )
