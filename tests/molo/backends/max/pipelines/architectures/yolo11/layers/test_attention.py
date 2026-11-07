import unittest
from ultralytics.nn.modules.block import Attention as Reference
from molo.backends.max.pipelines.architectures.yolo11.layers.attention import Attention
from tests.helpers import REF, compare


class AttentionTest(unittest.TestCase):
    def test_heads_and_ratio(self):
        for channels, heads, ratio, shape in [
            (64, 1, 0.5, (1, 64, 2, 3)),
            (64, 2, 0.5, (2, 64, 3, 2)),
            (64, 4, 0.25, (1, 64, 1, 2)),
        ]:
            with self.subTest(heads=heads, ratio=ratio):
                compare(
                    Attention(channels, heads, ratio, device=REF),
                    Reference(channels, num_heads=heads, attn_ratio=ratio),
                    shape,
                )

    def test_invalid_heads(self):
        with self.assertRaises(ValueError):
            Attention(64, num_heads=0, device=REF)
        with self.assertRaises(ValueError):
            Attention(64, num_heads=3, device=REF)
