"""Tests for the shared detection head."""

import unittest
from max.graph import DeviceRef
from molo.backends.max.pipelines.architectures.yolo.layers.detect import Detect


class DetectTest(unittest.TestCase):
    def test_invalid_inputs(self):
        for classes, channels, strides in (
            (0, (8,), (8,)),
            (2, (), ()),
            (2, (8,), (8, 16)),
        ):
            with self.subTest(classes=classes, channels=channels, strides=strides):
                with self.assertRaises(ValueError):
                    Detect(classes, channels, strides=strides, device=DeviceRef.CPU())
