"""Tests for yolo11 architecture registration."""

import unittest
from molo.backends.max.pipelines.architectures import get_architecture
from molo.backends.max.pipelines.architectures.yolo11.model import (
    build_graph,
)
from molo.backends.max.pipelines.architectures.yolo11.model import convert_model
from molo.backends.max.pipelines.architectures.yolo11.model import YOLO11_SIZES


class RegistrationTest(unittest.TestCase):
    def test_registration(self):
        for size in YOLO11_SIZES:
            with self.subTest(size=size):
                architecture = get_architecture(f"yolo11{size}")
                self.assertIs(architecture.build_graph, build_graph)
                self.assertIs(architecture.convert_model, convert_model)
