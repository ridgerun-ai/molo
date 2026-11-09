# Copyright (C) 2026 RidgeRun, LLC (http://www.ridgerun.com)
# SPDX-License-Identifier: AGPL-3.0-only

"""YOLO channel concatenation layer."""

from max.graph import TensorValue, ops
from max.nn import Module


class Concat(Module):
    """Concatenate input features along their channel dimension."""

    def __call__(self, inputs: list[TensorValue]) -> TensorValue:
        return ops.concat(inputs, axis=1)
