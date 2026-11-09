# Copyright (C) 2026 RidgeRun, LLC (http://www.ridgerun.com)
# SPDX-License-Identifier: AGPL-3.0-only

"""YOLO11 SPPF module for MAX Graph inference."""

from max.graph import DeviceRef, TensorValue, ops
from max.nn import Module

from .conv import Conv


class MaxPool2d(Module):
    """Stride-one NCHW max pooling with symmetric padding."""

    def __init__(self, kernel_size: int) -> None:
        super().__init__()
        self.kernel_size = kernel_size

    def __call__(self, x: TensorValue) -> TensorValue:
        pooled = ops.max_pool2d(
            ops.permute(x, [0, 2, 3, 1]),
            (self.kernel_size, self.kernel_size),
            stride=1,
            padding=self.kernel_size // 2,
        )
        return ops.permute(pooled, [0, 3, 1, 2])


class SPPF(Module):
    """Spatial pyramid pooling with three sequential max-pooling operations.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Size of each max-pooling kernel.
        device: Target device.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        device: DeviceRef = DeviceRef.GPU(),
    ) -> None:
        super().__init__()
        hidden = in_channels // 2
        self.cv1 = Conv(in_channels, hidden, device=device)
        self.cv2 = Conv(hidden * 4, out_channels, device=device)
        self.m = MaxPool2d(kernel_size)

    def __call__(self, x: TensorValue) -> TensorValue:
        """Pool sequentially, concatenate all four scales, and fuse them."""
        y = self.cv1(x)
        features = [y]
        for _ in range(3):
            features.append(self.m(features[-1]))
        return self.cv2(ops.concat(features, axis=1))
