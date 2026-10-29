"""Distribution focal loss integral for MAX Graph inference."""

import numpy as np
from max.dtype import DType
from max.graph import DeviceRef, TensorValue, ops
from max.nn import Module


class DFL(Module):
    """Decode four distributions into expected distances.

    Args:
        bins: Number of bins per distance; YOLO11 uses 16.
    """

    def __init__(self, bins: int = 16) -> None:
        super().__init__()
        if bins <= 0:
            raise ValueError("bins must be positive")
        self.bins = bins

    def __call__(self, x: TensorValue) -> TensorValue:
        """Decode input shaped (batch, 4 * bins, locations)."""
        batch, _, locations = x.shape
        logits = ops.reshape(x, (batch, 4, self.bins, locations))
        logits = ops.permute(logits, [0, 1, 3, 2])
        probabilities = ops.softmax(logits, axis=-1)
        bins = (
            ops.constant(
                np.arange(self.bins, dtype=np.float32).reshape(1, 1, 1, self.bins),
                dtype=DType.float32,
                device=DeviceRef.CPU(),
            )
            .to(x.device)
            .cast(x.dtype)
        )
        return ops.reshape(ops.sum(probabilities * bins, axis=3), (batch, 4, locations))
