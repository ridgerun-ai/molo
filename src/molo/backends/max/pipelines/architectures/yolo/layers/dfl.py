"""Distribution focal loss integral for MAX inference."""

from max.dtype import DType
from max.graph import DeviceRef, TensorValue, ops
from max.nn import Conv2d, Module


class DFL(Module):
    """Decode four distributions with the checkpoint's fixed projection."""

    def __init__(self, bins: int = 16, device: DeviceRef = DeviceRef.CPU()) -> None:
        super().__init__()
        if bins <= 0:
            raise ValueError("bins must be positive")
        self.bins = bins
        self.conv = Conv2d(
            1, bins, 1, DType.float32, has_bias=False, permute=True, device=device
        )

    def __call__(self, x: TensorValue) -> TensorValue:
        batch, _, locations = x.shape
        logits = ops.reshape(x, (batch, 4, self.bins, locations))
        logits = ops.permute(logits, [0, 1, 3, 2])
        probabilities = ops.softmax(logits, axis=-1)
        probabilities = ops.permute(probabilities, [0, 3, 1, 2])
        return ops.reshape(self.conv(probabilities), (batch, 4, locations))
