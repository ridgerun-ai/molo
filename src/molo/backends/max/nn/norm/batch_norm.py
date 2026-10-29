"""Batch normalization for MAX Graph inference."""

from __future__ import annotations

from dataclasses import dataclass

from max.dtype import DType
from max.graph import DeviceRef, TensorValue, Weight, ops
from max.nn import Module


@dataclass
class BatchNorm(Module):
    """Batch normalization block for inference.

    Normalizes each channel using stored running statistics and optionally
    applies a per-channel affine transformation. The input has shape
    ``(N, C, *)``; the output has the same shape.

    Args:
        num_channels: The number of input channels.
        eps: A small value added to the variance for numerical stability.
        affine: Whether to apply a per-channel scale and bias.
        device: The target DeviceRef for the weights.

    Note:
        This module implements inference only. Load ``running_mean`` and
        ``running_var`` from a checkpoint; when ``affine`` is true, also load
        ``weight`` and ``bias``. It does not update running statistics.
    """

    def __init__(
        self,
        num_channels: int,
        eps: float = 1e-5,
        affine: bool = True,
        device: DeviceRef = DeviceRef.GPU(),
        name: str | None = None,
    ) -> None:
        super().__init__()
        if num_channels <= 0:
            raise ValueError("num_channels must be positive")
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        self.running_mean = Weight(
            name=f"{name}.running_mean" if name else "running_mean",
            shape=(num_channels,),
            dtype=DType.float32,
            device=device,
        )
        self.running_var = Weight(
            name=f"{name}.running_var" if name else "running_var",
            shape=(num_channels,),
            dtype=DType.float32,
            device=device,
        )
        self.weight: Weight | None = None
        self.bias: Weight | None = None
        if affine:
            self.weight = Weight(
                name=f"{name}.weight" if name else "weight",
                shape=(num_channels,),
                dtype=DType.float32,
                device=device,
            )
            self.bias = Weight(
                name=f"{name}.bias" if name else "bias",
                shape=(num_channels,),
                dtype=DType.float32,
                device=device,
            )

    def __call__(self, x: TensorValue) -> TensorValue:
        """Normalize an ``(N, C, *)`` input using running statistics."""
        if len(x.shape) < 2:
            raise ValueError(
                f"Expected input tensor with at least 2 dimensions, got {x.shape}"
            )
        if x.shape[1] != self.num_channels:
            raise ValueError(
                f"Expected {self.num_channels} channels, got shape {x.shape}"
            )

        # Compute in float32 to preserve the checkpoint statistics for FP16 inputs.
        value = x.cast(DType.float32)
        shape = (1, self.num_channels) + (1,) * (len(x.shape) - 2)
        mean = ops.reshape(self.running_mean.to(x.device), shape)
        variance = ops.reshape(self.running_var.to(x.device), shape)
        result = (value - mean) * ops.rsqrt(variance + self.eps)
        if self.weight is not None and self.bias is not None:
            scale = ops.reshape(self.weight.to(x.device), shape)
            offset = ops.reshape(self.bias.to(x.device), shape)
            result = result * scale + offset
        return result.cast(x.dtype)
