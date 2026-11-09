# Copyright (C) 2026 RidgeRun, LLC (http://www.ridgerun.com)
# SPDX-License-Identifier: AGPL-3.0-only

"""Tests for yolov8 architecture registration."""

import unittest
from molo.backends.max.pipelines.architectures import get_architecture
from molo.backends.max.pipelines.architectures.yolov8.model import (
    build_graph,
)
from molo.backends.max.pipelines.architectures.yolov8.model import convert_model
from molo.backends.max.pipelines.architectures.yolov8.model import YOLOV8_SIZES


class RegistrationTest(unittest.TestCase):
    def test_registration(self):
        for size in YOLOV8_SIZES:
            with self.subTest(size=size):
                architecture = get_architecture(f"yolov8{size}")
                self.assertIs(architecture.build_graph, build_graph)
                self.assertIs(architecture.convert_model, convert_model)
