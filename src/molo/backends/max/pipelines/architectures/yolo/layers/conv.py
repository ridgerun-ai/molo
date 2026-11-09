# Copyright (C) 2026 RidgeRun, LLC (http://www.ridgerun.com)
# SPDX-License-Identifier: AGPL-3.0-only

"""YOLO-style convolution block for MAX Graph inference."""

from max.dtype import DType
from max.graph import DeviceRef, TensorValue, ops
from max.nn import Conv2d, Identity, Module


class SiLU(Module):
    """SiLU activation."""

    def __call__(self, x: TensorValue) -> TensorValue:
        return ops.silu(x)


class Conv(Module):
    """Fused convolution with bias and optional SiLU.

    Inputs and outputs use channel-first ``(N, C, H, W)`` layout. Convolution
    weights use PyTorch's ``(out_channels, in_channels / groups, H, W)`` layout.
    Batch normalization is folded into the convolution during conversion.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Convolution kernel size.
        stride: Convolution stride.
        padding: Convolution padding; defaults to automatic symmetric padding.
        groups: Number of convolution groups.
        dilation: Convolution dilation.
        activation: Whether to apply SiLU.
        device: Target device.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int | None = None,
        groups: int = 1,
        dilation: int = 1,
        activation: bool = True,
        device: DeviceRef = DeviceRef.GPU(),
    ) -> None:
        super().__init__()
        if padding is None:
            padding = (dilation * (kernel_size - 1) + 1) // 2
        self.conv = Conv2d(
            kernel_size=kernel_size,
            in_channels=in_channels,
            out_channels=out_channels,
            dtype=DType.float32,
            stride=stride,
            padding=padding,
            dilation=dilation,
            num_groups=groups,
            device=device,
            has_bias=True,
            permute=True,
        )
        self.act = SiLU() if activation else Identity()

    def __call__(self, x: TensorValue) -> TensorValue:
        """Apply fused convolution followed by optional SiLU."""
        x = self.conv(x)
        return self.act(x)
