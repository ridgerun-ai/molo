# Copyright (C) 2026 RidgeRun, LLC (http://www.ridgerun.com)
# SPDX-License-Identifier: AGPL-3.0-only

"""YOLO11 C2PSA module for MAX Graph inference."""

from max.graph import DeviceRef, TensorValue, ops
from max.nn import LayerList, Module
from ...yolo.layers.conv import Conv
from .psa_block import PSABlock


class C2PSA(Module):
    """Apply PSA blocks to one half of a split feature tensor.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels; must equal in_channels.
        repeats: Number of sequential PSA blocks.
        expansion: Ratio of hidden channels to input channels.
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
        if in_channels != out_channels:
            raise ValueError("in_channels must equal out_channels")
        self.hidden_channels = int(in_channels * expansion)
        c = self.hidden_channels
        heads = c // 64
        if heads < 1:
            raise ValueError("hidden channels must be at least 64")
        self.cv1 = Conv(in_channels, 2 * c, device=device)
        self.cv2 = Conv(2 * c, out_channels, device=device)
        self.m = LayerList(
            [PSABlock(c, num_heads=heads, device=device) for i in range(repeats)]
        )

    def __call__(self, x: TensorValue) -> TensorValue:
        """Split, process one path, concatenate, and fuse."""
        c = self.hidden_channels
        a, b = ops.split(self.cv1(x), [c, c], axis=1)
        for block in self.m:
            b = block(b)
        return self.cv2(ops.concat((a, b), axis=1))
