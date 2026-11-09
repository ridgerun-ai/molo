# Copyright (C) 2026 RidgeRun, LLC (http://www.ridgerun.com)
# SPDX-License-Identifier: AGPL-3.0-only

"""YOLO11 position-sensitive attention block for MAX Graph inference."""

from max.graph import DeviceRef, TensorValue
from max.nn import Module, Sequential
from .attention import Attention
from ...yolo.layers.conv import Conv


class PSABlock(Module):
    """Spatial attention followed by a feed-forward network.

    Args:
        channels: Number of input and output channels.
        attention_ratio: Query and key dimension as a fraction of head dimension.
        num_heads: Number of attention heads.
        shortcut: Add residual connections around attention and feed-forward stages.
        device: Target device.
    """

    def __init__(
        self,
        channels: int,
        attention_ratio: float = 0.5,
        num_heads: int = 4,
        shortcut: bool = True,
        device: DeviceRef = DeviceRef.GPU(),
    ) -> None:
        super().__init__()
        self.attn = Attention(channels, num_heads, attention_ratio, device)
        self.ffn = Sequential(
            [
                Conv(channels, 2 * channels, device=device),
                Conv(2 * channels, channels, activation=False, device=device),
            ]
        )
        self.add = shortcut

    def __call__(self, x: TensorValue) -> TensorValue:
        """Apply attention and feed-forward stages."""
        attention = self.attn(x)
        x = x + attention if self.add else attention
        feed_forward = self.ffn(x)
        return x + feed_forward if self.add else feed_forward
