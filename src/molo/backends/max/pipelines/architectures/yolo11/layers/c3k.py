"""YOLO C3k module for MAX Graph inference."""

from max.graph import DeviceRef, TensorValue, ops
from max.nn import LayerList, Module

from ...yolo.layers.bottleneck import Bottleneck
from ...yolo.layers.conv import Conv


class C3k(Module):
    """Two-path CSP block with sequential bottlenecks.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        repeats: Number of sequential bottlenecks.
        shortcut: Enable bottleneck residual connections.
        groups: Number of groups in the second bottleneck convolution.
        expansion: Ratio of hidden channels to output channels.
        kernel_size: Kernel size for both bottleneck convolutions.
        device: Target device.
        name: Optional prefix for weight names.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        repeats: int = 1,
        shortcut: bool = True,
        groups: int = 1,
        expansion: float = 0.5,
        kernel_size: int = 3,
        device: DeviceRef = DeviceRef.GPU(),
        name: str | None = None,
    ) -> None:
        super().__init__()
        hidden = int(out_channels * expansion)
        prefix = f"{name}." if name else ""
        self.cv1 = Conv(in_channels, hidden, device=device, name=f"{prefix}cv1")
        self.cv2 = Conv(in_channels, hidden, device=device, name=f"{prefix}cv2")
        self.cv3 = Conv(2 * hidden, out_channels, device=device, name=f"{prefix}cv3")
        self.m = LayerList(
            [
                Bottleneck(
                    hidden,
                    hidden,
                    shortcut=shortcut,
                    groups=groups,
                    kernel_size=(kernel_size, kernel_size),
                    expansion=1.0,
                    device=device,
                    name=f"{prefix}m.{i}",
                )
                for i in range(repeats)
            ]
        )

    def __call__(self, x: TensorValue) -> TensorValue:
        """Apply both paths, concatenate channels, and fuse them."""
        y = self.cv1(x)
        for block in self.m:
            y = block(y)
        return self.cv3(ops.concat((y, self.cv2(x)), axis=1))
