# Copyright (C) 2026 RidgeRun, LLC (http://www.ridgerun.com)
# SPDX-License-Identifier: AGPL-3.0-only

import unittest
import torch
from ultralytics.nn.modules.block import SPPF as Reference
from molo.backends.max.pipelines.architectures.yolo.layers.sppf import SPPF
from tests.helpers import REF, compare


class SPPFTest(unittest.TestCase):
    def test_yolo11_configurations(self):
        for ci, co, kernel, spatial in [
            (8, 8, 5, (5, 7)),
            (6, 10, 3, (3, 4)),
            (8, 8, 5, (1, 1)),
        ]:
            with self.subTest(kernel=kernel, spatial=spatial):
                reference = Reference(ci, co, k=kernel)
                # Newer Ultralytics SPPF defaults to no activation. YOLO11's
                # original SPPF checkpoint uses SiLU in this position.
                reference.cv1.act = torch.nn.SiLU()
                compare(
                    SPPF(ci, co, kernel, device=REF),
                    reference,
                    (1, ci, *spatial),
                )
