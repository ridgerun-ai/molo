"""Test YOLO11 artifact size metadata."""

import json
import unittest
from unittest.mock import patch

import numpy as np

from molo.backends.max.pipelines.architectures.yolo11.artifact import load_artifact


class Archive(dict):
    @property
    def files(self):
        return list(self)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class ArtifactTest(unittest.TestCase):
    def test_sizes(self):
        for size in ("n", "s", "m", "l", "x"):
            with self.subTest(size=size):
                metadata = {
                    "model_id": f"yolo11{size}",
                    "names": {"0": "object"},
                }
                archive = Archive(
                    __molo_metadata__=np.frombuffer(
                        json.dumps(metadata).encode(), dtype=np.uint8
                    ),
                    weight=np.array([1], dtype=np.float32),
                )
                with patch(
                    "molo.backends.max.pipelines.architectures.yolo11.artifact.np.load",
                    return_value=archive,
                ):
                    weights, names, actual_size = load_artifact("model.max")
                self.assertEqual(actual_size, size)
                self.assertEqual(names, {0: "object"})
                np.testing.assert_array_equal(weights["weight"], [1])

    def test_missing_or_invalid_identifier(self):
        for model_id in (None, "invalid"):
            with self.subTest(model_id=model_id):
                metadata = {"model_id": model_id, "names": {"0": "object"}}
                archive = Archive(
                    __molo_metadata__=np.frombuffer(
                        json.dumps(metadata).encode(), dtype=np.uint8
                    )
                )
                with patch(
                    "molo.backends.max.pipelines.architectures.yolo11.artifact.np.load",
                    return_value=archive,
                ):
                    with self.assertRaisesRegex(
                        ValueError, "Unsupported model identifier"
                    ):
                        load_artifact("model.max")
