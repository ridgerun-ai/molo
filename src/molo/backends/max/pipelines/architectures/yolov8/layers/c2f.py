# Copyright (C) 2026 RidgeRun, LLC (http://www.ridgerun.com)
# SPDX-License-Identifier: AGPL-3.0-only

"""YOLOv8 C2f block."""

from max.graph import DeviceRef, TensorValue, ops
from max.nn import LayerList, Module

from ...yolo.layers.bottleneck import Bottleneck
from ...yolo.layers.conv import Conv


class C2f(Module):
    """Split, process and concatenate features before a final convolution."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        repeats: int = 1,
        shortcut: bool = False,
        groups: int = 1,
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
                Bottleneck(
                    c,
                    c,
                    shortcut=shortcut,
                    groups=groups,
                    expansion=1.0,
                    device=device,
                )
                for i in range(repeats)
            ]
        )

    def __call__(self, x: TensorValue) -> TensorValue:
        c = self.hidden_channels
        parts = ops.split(self.cv1(x), [c, c], axis=1)
        for block in self.m:
            parts.append(block(parts[-1]))
        return self.cv2(ops.concat(parts, axis=1))
