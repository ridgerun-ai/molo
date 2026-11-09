# Copyright (C) 2026 RidgeRun, LLC (http://www.ridgerun.com)
# SPDX-License-Identifier: AGPL-3.0-only

"""YOLO26 residual spatial pyramid pooling."""

from max.graph import DeviceRef, TensorValue, ops
from max.nn import Module

from ...yolo.layers.conv import Conv
from ...yolo.layers.sppf import MaxPool2d


class SPPF(Module):
    """Pool a linear projection at three scales and add the input residual.

    Args:
        channels: Input and output channels.
        kernel_size: Size of each max-pooling kernel.
        device: Target device.
    """

    def __init__(
        self, channels: int, kernel_size: int = 5, device: DeviceRef = DeviceRef.GPU()
    ) -> None:
        super().__init__()
        hidden = channels // 2
        self.cv1 = Conv(channels, hidden, activation=False, device=device)
        self.cv2 = Conv(hidden * 4, channels, device=device)
        self.m = MaxPool2d(kernel_size)

    def __call__(self, x: TensorValue) -> TensorValue:
        """Fuse the pooled features and apply the residual connection."""
        features = [self.cv1(x)]
        for _ in range(3):
            features.append(self.m(features[-1]))
        return self.cv2(ops.concat(features, axis=1)) + x
