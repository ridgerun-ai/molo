"""YOLO11 spatial attention for MAX Graph inference."""

from max.graph import DeviceRef, TensorValue, ops
from max.nn import Module
from .conv import Conv


class Attention(Module):
    """Multi-head spatial attention with depthwise positional encoding.

    Args:
        channels: Number of input and output channels.
        num_heads: Number of attention heads.
        attention_ratio: Key and query dimension as a fraction of head dimension.
        device: Target device.
        name: Optional prefix for weight names.
    """

    def __init__(
        self,
        channels: int,
        num_heads: int = 8,
        attention_ratio: float = 0.5,
        device: DeviceRef = DeviceRef.GPU(),
        name: str | None = None,
    ) -> None:
        super().__init__()
        if num_heads <= 0 or channels % num_heads:
            raise ValueError("channels must be divisible by a positive num_heads")
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.key_dim = int(self.head_dim * attention_ratio)
        if self.key_dim <= 0:
            raise ValueError("attention_ratio must produce a positive key dimension")
        self.scale = self.key_dim**-0.5
        prefix = f"{name}." if name else ""
        self.qkv = Conv(
            channels,
            channels + 2 * num_heads * self.key_dim,
            activation=False,
            device=device,
            name=f"{prefix}qkv",
        )
        self.proj = Conv(
            channels, channels, activation=False, device=device, name=f"{prefix}proj"
        )
        self.pe = Conv(
            channels,
            channels,
            kernel_size=3,
            groups=channels,
            activation=False,
            device=device,
            name=f"{prefix}pe",
        )

    def __call__(self, x: TensorValue) -> TensorValue:
        """Apply attention over spatial positions."""
        batch, _, height, width = x.shape
        qkv = ops.reshape(
            self.qkv(x), (batch, self.num_heads, 2 * self.key_dim + self.head_dim, -1)
        )
        q, k, v = ops.split(qkv, [self.key_dim, self.key_dim, self.head_dim], axis=2)
        scores = ops.matmul(ops.transpose(q, -2, -1), k) * self.scale
        attention = ops.softmax(scores, axis=-1)
        output = ops.matmul(v, ops.transpose(attention, -2, -1))
        shape = (batch, self.channels, height, width)
        output = ops.reshape(output, shape) + self.pe(ops.reshape(v, shape))
        return self.proj(output)
