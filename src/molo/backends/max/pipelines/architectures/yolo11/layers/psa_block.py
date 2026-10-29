"""YOLO11 position-sensitive attention block for MAX Graph inference."""

from max.graph import DeviceRef, TensorValue
from max.nn import Module
from .attention import Attention
from .conv import Conv


class PSABlock(Module):
    """Spatial attention followed by a feed-forward network.

    Args:
        channels: Number of input and output channels.
        attention_ratio: Query and key dimension as a fraction of head dimension.
        num_heads: Number of attention heads.
        shortcut: Add residual connections around attention and feed-forward stages.
        device: Target device.
        name: Optional prefix for weight names.
    """

    def __init__(
        self,
        channels: int,
        attention_ratio: float = 0.5,
        num_heads: int = 4,
        shortcut: bool = True,
        device: DeviceRef = DeviceRef.GPU(),
        name: str | None = None,
    ) -> None:
        super().__init__()
        prefix = f"{name}." if name else ""
        self.attn = Attention(
            channels, num_heads, attention_ratio, device, f"{prefix}attn"
        )
        self.ffn1 = Conv(channels, 2 * channels, device=device, name=f"{prefix}ffn.0")
        self.ffn2 = Conv(
            2 * channels,
            channels,
            activation=False,
            device=device,
            name=f"{prefix}ffn.1",
        )
        self.add = shortcut

    def __call__(self, x: TensorValue) -> TensorValue:
        """Apply attention and feed-forward stages."""
        attention = self.attn(x)
        x = x + attention if self.add else attention
        feed_forward = self.ffn2(self.ffn1(x))
        return x + feed_forward if self.add else feed_forward
