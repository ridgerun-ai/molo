import unittest
from ultralytics.nn.modules.block import C3k2 as Reference
from molo.backends.max.pipelines.architectures.yolo11.layers.c3k2 import C3k2
from tests.unit.helpers import REF, compare


class C3k2Test(unittest.TestCase):
    def test_configurations(self):
        for repeats, c3k, expansion, shortcut in [
            (1, False, 0.5, True),
            (2, False, 0.25, False),
            (1, True, 0.5, True),
            (2, True, 1.0, False),
        ]:
            with self.subTest(
                repeats=repeats, c3k=c3k, expansion=expansion, shortcut=shortcut
            ):
                compare(
                    C3k2(
                        8,
                        8,
                        repeats,
                        c3k,
                        expansion,
                        shortcut=shortcut,
                        device=REF,
                        name="layer",
                    ),
                    Reference(8, 8, n=repeats, c3k=c3k, e=expansion, shortcut=shortcut),
                    (1, 8, 3, 3),
                )
