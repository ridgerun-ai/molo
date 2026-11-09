# Copyright (C) 2026 RidgeRun, LLC (http://www.ridgerun.com)
# SPDX-License-Identifier: AGPL-3.0-only

import unittest
from ultralytics.nn.modules.block import C3k as Reference
from molo.backends.max.pipelines.architectures.yolo11.layers.c3k import C3k
from tests.helpers import REF, compare


class C3kTest(unittest.TestCase):
    def test_configurations(self):
        for repeats, shortcut, groups, expansion in [
            (1, True, 1, 0.5),
            (2, True, 1, 0.5),
            (2, False, 2, 1.0),
        ]:
            with self.subTest(repeats=repeats, shortcut=shortcut, groups=groups):
                compare(
                    C3k(
                        8,
                        8,
                        repeats,
                        shortcut,
                        groups,
                        expansion,
                        device=REF,
                    ),
                    Reference(
                        8, 8, n=repeats, shortcut=shortcut, g=groups, e=expansion
                    ),
                    (1, 8, 5, 5),
                )
