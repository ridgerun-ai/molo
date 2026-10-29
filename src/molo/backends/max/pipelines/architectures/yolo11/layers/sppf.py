"""YOLO11 SPPF module for MAX Graph inference."""

from max.graph import DeviceRef, TensorValue, ops
from max.nn import Module

from .conv import Conv


class SPPF(Module):
    """Spatial pyramid pooling with three sequential max-pooling operations.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Size of each max-pooling kernel.
        device: Target device.
        name: Optional prefix for weight names.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        device: DeviceRef = DeviceRef.GPU(),
        name: str | None = None,
    ) -> None:
        super().__init__()
        hidden = in_channels // 2
        prefix = f"{name}." if name else ""
        self.cv1 = Conv(in_channels, hidden, device=device, name=f"{prefix}cv1")
        self.cv2 = Conv(hidden * 4, out_channels, device=device, name=f"{prefix}cv2")
        self.kernel_size = kernel_size

    def __call__(self, x: TensorValue) -> TensorValue:
        """Pool sequentially, concatenate all four scales, and fuse them."""
        y = self.cv1(x)
        # MAX max_pool2d takes NHWC; the surrounding convolutions use NCHW.
        pooled = ops.permute(y, [0, 2, 3, 1])
        features = [y]
        for _ in range(3):
            pooled = ops.max_pool2d(
                pooled,
                (self.kernel_size, self.kernel_size),
                stride=1,
                padding=self.kernel_size // 2,
            )
            features.append(ops.permute(pooled, [0, 3, 1, 2]))
        return self.cv2(ops.concat(features, axis=1))
