import unittest
from ultralytics.nn.modules.block import PSABlock as Reference
from molo.backends.max.pipelines.architectures.yolo11.layers.psa_block import PSABlock
from tests.helpers import REF, compare


class PSABlockTest(unittest.TestCase):
    def test_shortcuts(self):
        for shortcut, heads in [(True, 2), (False, 1)]:
            with self.subTest(shortcut=shortcut, heads=heads):
                compare(
                    PSABlock(64, num_heads=heads, shortcut=shortcut, device=REF),
                    Reference(64, num_heads=heads, shortcut=shortcut),
                    (1, 64, 2, 3),
                )
