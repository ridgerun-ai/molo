"""Tests for the YOLO concatenation layer."""

import unittest
from typing import cast
from unittest.mock import Mock, patch

from max.graph import TensorValue

from molo.backends.max.pipelines.architectures.yolo.layers.concat import Concat


class ConcatTest(unittest.TestCase):
    def test_concatenates_channels(self):
        for input_count in (2, 3):
            with self.subTest(input_count=input_count):
                inputs = [
                    cast(TensorValue, Mock(spec=TensorValue))
                    for _ in range(input_count)
                ]
                with patch(
                    "molo.backends.max.pipelines.architectures.yolo.layers.concat.ops.concat"
                ) as concat:
                    output = Concat()(inputs)

                concat.assert_called_once_with(inputs, axis=1)
                self.assertIs(concat.call_args.args[0], inputs)
                self.assertIs(output, concat.return_value)
