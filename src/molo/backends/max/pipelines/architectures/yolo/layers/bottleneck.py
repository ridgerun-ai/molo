# Copyright (C) 2026 RidgeRun, LLC (http://www.ridgerun.com)
# SPDX-License-Identifier: AGPL-3.0-only

"""YOLO bottleneck module for MAX Graph inference."""

from max.graph import DeviceRef, TensorValue
from max.nn import Module

from .conv import Conv


class Bottleneck(Module):
    """Two convolution blocks with an optional residual connection.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        shortcut: Add the input when input and output channels match.
        groups: Number of groups in the second convolution.
        kernel_size: Kernel sizes for the first and second convolutions.
        expansion: Ratio of hidden channels to output channels.
        device: Target device.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        shortcut: bool = True,
        groups: int = 1,
        kernel_size: tuple[int, int] = (3, 3),
        expansion: float = 0.5,
        device: DeviceRef = DeviceRef.GPU(),
    ) -> None:
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.cv1 = Conv(
            in_channels,
            hidden_channels,
            kernel_size=kernel_size[0],
            device=device,
        )
        self.cv2 = Conv(
            hidden_channels,
            out_channels,
            kernel_size=kernel_size[1],
            groups=groups,
            device=device,
        )
        self.add = shortcut and in_channels == out_channels

    def __call__(self, x: TensorValue) -> TensorValue:
        """Apply the bottleneck and its optional shortcut."""
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y
