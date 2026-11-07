"""Tests for YOLO26 architecture registration and public exports."""

import unittest

from molo.backends.max.pipelines.architectures import get_architecture
from molo.backends.max.pipelines.architectures.yolo26 import YOLO26
from molo.backends.max.pipelines.architectures.yolo26.model import (
    YOLO26_SIZES,
    YOLO26 as Model,
    build_graph,
    convert_model,
)


class RegistrationTest(unittest.TestCase):
    def test_registration(self):
        for size in YOLO26_SIZES:
            with self.subTest(size=size):
                architecture = get_architecture(f"yolo26{size}")
                self.assertIs(architecture.build_graph, build_graph)
                self.assertIs(architecture.convert_model, convert_model)

    def test_public_model(self):
        self.assertIs(YOLO26, Model)
