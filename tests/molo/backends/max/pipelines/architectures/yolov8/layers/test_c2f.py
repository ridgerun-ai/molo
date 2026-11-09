# Copyright (C) 2026 RidgeRun, LLC (http://www.ridgerun.com)
# SPDX-License-Identifier: AGPL-3.0-only

import unittest

from ultralytics.nn.modules.block import C2f as Reference

from molo.backends.max.pipelines.architectures.yolov8.layers.c2f import C2f
from tests.helpers import REF, compare


class C2fTest(unittest.TestCase):
    def test_configurations(self):
        for repeats, shortcut, expansion in [
            (1, False, 0.5),
            (2, True, 0.5),
            (2, False, 1.0),
        ]:
            with self.subTest(repeats=repeats, shortcut=shortcut, expansion=expansion):
                compare(
                    C2f(
                        8,
                        8,
                        repeats=repeats,
                        shortcut=shortcut,
                        expansion=expansion,
                        device=REF,
                    ),
                    Reference(8, 8, n=repeats, shortcut=shortcut, e=expansion),
                    (1, 8, 3, 3),
                )
