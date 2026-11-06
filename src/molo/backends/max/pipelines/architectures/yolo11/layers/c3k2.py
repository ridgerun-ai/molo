"""YOLO11 C3k2 module for MAX Graph inference."""

from max.graph import DeviceRef, TensorValue, ops
from max.nn import Module

from ...yolo.layers.bottleneck import Bottleneck
from .c3k import C3k
from ...yolo.layers.conv import Conv


class C3k2(Module):
    """Split, process and concatenate features before a final convolution.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        repeats: Number of sequential internal blocks.
        c3k: Use C3k blocks instead of bottlenecks.
        expansion: Ratio of hidden channels to output channels.
        groups: Number of groups in internal bottleneck convolutions.
        shortcut: Enable internal bottleneck residual connections.
        device: Target device.
        name: Optional prefix for weight names.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        repeats: int = 1,
        c3k: bool = False,
        expansion: float = 0.5,
        groups: int = 1,
        shortcut: bool = True,
        device: DeviceRef = DeviceRef.GPU(),
        name: str | None = None,
    ) -> None:
        super().__init__()
        self.hidden_channels = int(out_channels * expansion)
        prefix = f"{name}." if name else ""
        c = self.hidden_channels
        self.cv1 = Conv(in_channels, 2 * c, device=device, name=f"{prefix}cv1")
        self.cv2 = Conv(
            (2 + repeats) * c, out_channels, device=device, name=f"{prefix}cv2"
        )
        blocks = []
        for i in range(repeats):
            if c3k:
                block = C3k(
                    c,
                    c,
                    repeats=2,
                    shortcut=shortcut,
                    groups=groups,
                    device=device,
                    name=f"{prefix}m.{i}",
                )
            else:
                block = Bottleneck(
                    c,
                    c,
                    shortcut=shortcut,
                    groups=groups,
                    device=device,
                    name=f"{prefix}m.{i}",
                )
            blocks.append(block)
        self.m = blocks

    def __call__(self, x: TensorValue) -> TensorValue:
        """Apply the split paths and fuse all intermediate outputs."""
        c = self.hidden_channels
        parts = ops.split(self.cv1(x), [c, c], axis=1)
        for block in self.m:
            parts.append(block(parts[-1]))
        return self.cv2(ops.concat(parts, axis=1))
