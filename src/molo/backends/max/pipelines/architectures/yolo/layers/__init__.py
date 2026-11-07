"""Shared YOLO graph layers."""

from max.graph import TensorValue, ops
from max.nn import Module


class Concat(Module):
    """Concatenate input features along their channel dimension."""

    def __call__(self, inputs: list[TensorValue]) -> TensorValue:
        return ops.concat(inputs, axis=1)
