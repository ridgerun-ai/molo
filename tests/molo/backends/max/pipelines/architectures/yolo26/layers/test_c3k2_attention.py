"""Compare the YOLO26 attention block with Ultralytics."""

import unittest

from ultralytics.nn.modules.block import C3k2 as Reference
from molo.backends.max.pipelines.architectures.yolo26.layers.c3k2_attention import (
    C3k2Attention,
)
from tests.helpers import REF, compare


class C3k2AttentionTest(unittest.TestCase):
    def test_configurations(self):
        for ci, co, repeats, expansion, shape in [
            (32, 32, 1, 0.5, (1, 32, 3, 5)),
            (64, 128, 2, 0.5, (2, 64, 2, 3)),
            (128, 256, 1, 0.5, (1, 128, 2, 2)),
        ]:
            with self.subTest(channels=co, repeats=repeats, shape=shape):
                compare(
                    C3k2Attention(ci, co, repeats, expansion, device=REF),
                    Reference(ci, co, n=repeats, e=expansion, attn=True),
                    shape,
                )
