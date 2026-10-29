"""Compare YOLO11 size scaling with Ultralytics."""

import unittest

from max.graph import DeviceRef
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel

from molo.backends.max.pipelines.architectures.yolo11 import YOLO11, YOLO11Config


class ModelSizesTest(unittest.TestCase):
    def test_architecture_scaling(self):
        for size in ("n", "s", "m", "l", "x"):
            with self.subTest(size=size):
                model = YOLO11(
                    YOLO11Config(size=size, device=DeviceRef.CPU()), (64, 96)
                )
                reference = YOLO(f"yolo11{size}.yaml").model.model  # ty: ignore[unresolved-attribute]
                for i in (0, 1, 3, 5, 7, 17, 20):
                    actual = getattr(model, f"layer{i}").conv
                    expected = reference[i].conv
                    self.assertEqual(
                        (actual.in_channels, actual.out_channels),
                        (expected.in_channels, expected.out_channels),
                    )
                for i in (2, 4, 6, 8, 13, 16, 19, 22):
                    actual = getattr(model, f"layer{i}")
                    expected = reference[i]
                    self.assertEqual(len(actual.m), len(expected.m))
                    self.assertEqual(
                        actual.cv1.conv.out_channels, expected.cv1.conv.out_channels
                    )
                    self.assertEqual(
                        type(actual.m[0]).__name__, type(expected.m[0]).__name__
                    )
                self.assertEqual(len(model.layer10.m), len(reference[10].m))

    def test_custom_class_count_scaling(self):
        for size in ("n", "m", "x"):
            with self.subTest(size=size):
                yaml = YOLO(f"yolo11{size}.yaml").model.yaml  # ty: ignore[unresolved-attribute]
                reference = DetectionModel(yaml, ch=3, nc=64, verbose=False)
                model = YOLO11(YOLO11Config(size=size, nc=64), (64, 64))
                for i in (0, 1, 3, 5, 7, 17, 20):
                    actual = getattr(model, f"layer{i}").conv
                    expected = reference.model[i].conv
                    self.assertEqual(
                        (actual.in_channels, actual.out_channels),
                        (expected.in_channels, expected.out_channels),
                    )

    def test_invalid_size(self):
        with self.assertRaisesRegex(ValueError, "Unsupported YOLO11 size"):
            YOLO11(YOLO11Config(size="bad"), (64, 64))  # ty: ignore[invalid-argument-type]
