"""YOLO26 split-and-fuse block with position-sensitive attention."""

from max.graph import DeviceRef, TensorValue, ops
from max.nn import LayerList, Module, Sequential

from ...yolo.layers.bottleneck import Bottleneck
from ...yolo.layers.conv import Conv
from ...yolo11.layers.psa_block import PSABlock


class C3k2Attention(Module):
    """Fuse a bypass with repeated bottleneck and attention blocks.

    Args:
        in_channels: Input channels.
        out_channels: Output channels.
        repeats: Number of bottleneck/attention pairs.
        expansion: Hidden channels as a fraction of output channels.
        device: Target device.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        repeats: int = 1,
        expansion: float = 0.5,
        device: DeviceRef = DeviceRef.GPU(),
    ) -> None:
        super().__init__()
        self.hidden_channels = int(out_channels * expansion)
        c = self.hidden_channels
        self.cv1 = Conv(in_channels, 2 * c, device=device)
        self.cv2 = Conv((2 + repeats) * c, out_channels, device=device)
        self.m = LayerList(
            [
                Sequential(
                    [
                        Bottleneck(c, c, device=device),
                        PSABlock(
                            c,
                            attention_ratio=0.5,
                            num_heads=max(c // 64, 1),
                            device=device,
                        ),
                    ]
                )
                for _ in range(repeats)
            ]
        )

    def __call__(self, x: TensorValue) -> TensorValue:
        """Process the split path and fuse all intermediate features."""
        parts = ops.split(
            self.cv1(x), [self.hidden_channels, self.hidden_channels], axis=1
        )
        for block in self.m:
            parts.append(block(parts[-1]))
        return self.cv2(ops.concat(parts, axis=1))
