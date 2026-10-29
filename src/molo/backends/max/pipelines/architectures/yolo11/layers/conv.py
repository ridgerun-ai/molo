"""YOLO-style convolution block for MAX Graph inference."""

from max.dtype import DType
from max.graph import DeviceRef, TensorValue, ops
from max.nn import Conv2d, Module

from molo.backends.max.nn.norm.batch_norm import BatchNorm


class Conv(Module):
    """Convolution, batch normalization, and optional SiLU.

    Inputs and outputs use channel-first ``(N, C, H, W)`` layout. Convolution
    weights use PyTorch's ``(out_channels, in_channels / groups, H, W)`` layout.
    Batch normalization uses stored running statistics for inference.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Convolution kernel size.
        stride: Convolution stride.
        padding: Convolution padding; defaults to automatic symmetric padding.
        groups: Number of convolution groups.
        dilation: Convolution dilation.
        activation: Whether to apply SiLU.
        eps: Batch normalization epsilon.
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
        eps: float = 1e-3,
        device: DeviceRef = DeviceRef.GPU(),
        name: str | None = None,
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
            has_bias=False,
            permute=True,
            name=f"{name}.conv" if name else "conv",
        )
        self.bn = BatchNorm(
            out_channels, eps=eps, device=device, name=f"{name}.bn" if name else None
        )
        self.activation = activation

    def __call__(self, x: TensorValue) -> TensorValue:
        """Apply convolution and normalization, followed by optional SiLU."""
        x = self.bn(self.conv(x))
        return ops.silu(x) if self.activation else x
