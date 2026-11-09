# Copyright (C) 2026 RidgeRun, LLC (http://www.ridgerun.com)
# SPDX-License-Identifier: AGPL-3.0-only

"""Test YOLO11 artifact size metadata."""

import json
import unittest
from unittest.mock import patch

import numpy as np

from molo.backends.max.pipelines.architectures.yolo.artifact import load_artifact


class Archive(dict):
    @property
    def files(self):
        return list(self)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class ArtifactTest(unittest.TestCase):
    def test_model_identifiers(self):
        for model_id in (
            f"{family}{size}"
            for family in ("yolo11", "yolov8", "another_architecture")
            for size in ("n", "s", "m", "l", "x")
        ):
            with self.subTest(model_id=model_id):
                metadata = {
                    "model_id": model_id,
                    "class_names": ["object"],
                }
                archive = Archive(
                    __molo_metadata__=np.frombuffer(
                        json.dumps(metadata).encode(), dtype=np.uint8
                    ),
                    weight=np.array([1], dtype=np.float32),
                )
                with patch(
                    "molo.backends.max.pipelines.architectures.yolo.artifact.np.load",
                    return_value=archive,
                ):
                    artifact = load_artifact("model.max")
                self.assertEqual(artifact.model_id, model_id)
                self.assertEqual(artifact.class_names, ["object"])
                np.testing.assert_array_equal(artifact.weights["weight"], [1])
