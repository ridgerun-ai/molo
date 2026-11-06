import unittest
from ultralytics.nn.modules.block import C2PSA as Reference
from molo.backends.max.pipelines.architectures.yolo11.layers.c2psa import C2PSA
from tests.helpers import REF, compare


class C2PSATest(unittest.TestCase):
    def test_repeats(self):
        for repeats in (1, 2):
            with self.subTest(repeats=repeats):
                compare(
                    C2PSA(128, 128, repeats=repeats, device=REF, name="layer"),
                    Reference(128, 128, n=repeats),
                    (1, 128, 2, 3),
                )

    def test_mismatched_channels(self):
        with self.assertRaises(ValueError):
            C2PSA(128, 256, device=REF)
