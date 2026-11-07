from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import ceil

from max import nn
from max.dtype import DType
from max.graph import DeviceRef, Graph, TensorType, ops

from .layers.c2psa import C2PSA
from .layers.c3k2 import C3k2
from .layers.conv import Conv
from .layers.detect import Detect
from .layers.sppf import SPPF
from .layers.upsample import UpsampleNearest2x
from .model_config import YOLO11_SIZES, YOLO11Size


@dataclass(frozen=True)
class YOLO11Config:
    size: YOLO11Size = "n"
    nc: int = 80
    input_size: int = 640
    reg_max: int = 16
    dtype: DType = DType.float32
    device: DeviceRef = DeviceRef.CPU()


class YOLO11(nn.Module):
    """YOLO11 detection model, parameterized by size."""

    def __init__(self, config: YOLO11Config, input_shape: tuple[int, int]) -> None:
        super().__init__()
        self.config = config
        try:
            scale = YOLO11_SIZES[config.size]
        except KeyError as exc:
            raise ValueError(f"Unsupported YOLO11 size: {config.size!r}") from exc
        if config.dtype != DType.float32 or config.reg_max != 16:
            raise ValueError("YOLO11 currently supports float32 and reg_max=16")

        def channels(base: int) -> int:
            if base == config.nc:
                return base
            return ceil(min(base, scale.max_channels) * scale.width / 8) * 8

        repeats = max(round(2 * scale.depth), 1)
        c64, c128, c256, c512, c1024 = (channels(c) for c in (64, 128, 256, 512, 1024))
        device = config.device

        def down(ci: int, co: int, index: int) -> Conv:
            return Conv(
                ci, co, kernel_size=3, stride=2, device=device, name=f"layer{index}"
            )

        def block(
            ci: int, co: int, index: int, *, c3k: bool = False, expansion: float = 0.5
        ) -> C3k2:
            return C3k2(
                ci,
                co,
                repeats=repeats,
                c3k=c3k or scale.c3k,
                expansion=expansion,
                device=device,
                name=f"layer{index}",
            )

        self.layer0 = down(3, c64, 0)
        self.layer1 = down(c64, c128, 1)
        self.layer2 = block(c128, c256, 2, expansion=0.25)
        self.layer3 = down(c256, c256, 3)
        self.layer4 = block(c256, c512, 4, expansion=0.25)
        self.layer5 = down(c512, c512, 5)
        self.layer6 = block(c512, c512, 6, c3k=True)
        self.layer7 = down(c512, c1024, 7)
        self.layer8 = block(c1024, c1024, 8, c3k=True)
        self.layer9 = SPPF(c1024, c1024, kernel_size=5, device=device, name="layer9")
        self.layer10 = C2PSA(
            c1024, c1024, repeats=repeats, expansion=0.5, device=device, name="layer10"
        )
        self.upsample0 = UpsampleNearest2x()
        self.layer13 = block(c1024 + c512, c512, 13)
        self.upsample1 = UpsampleNearest2x()
        self.layer16 = block(c512 + c512, c256, 16)
        self.layer17 = down(c256, c256, 17)
        self.layer19 = block(c256 + c512, c512, 19)
        self.layer20 = down(c512, c512, 20)
        self.layer22 = block(c512 + c1024, c1024, 22, c3k=True)
        self.detect = Detect(
            num_classes=config.nc,
            in_channels=(c256, c512, c1024),
            device=device,
            name="detect",
        )

    def __call__(self, x):
        x0 = self.layer0(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x5 = self.layer5(x4)
        x6 = self.layer6(x5)
        x7 = self.layer7(x6)
        x8 = self.layer8(x7)
        x9 = self.layer9(x8)
        x10 = self.layer10(x9)

        x11 = self.upsample0(x10)
        x12 = self.layer13(ops.concat((x11, x6), axis=1))
        x13 = self.upsample1(x12)
        x14 = self.layer16(ops.concat((x13, x4), axis=1))
        x15 = self.layer17(x14)
        x16 = self.layer19(ops.concat((x15, x12), axis=1))
        x17 = self.layer20(x16)
        x18 = self.layer22(ops.concat((x17, x10), axis=1))
        return self.detect([x14, x16, x18])


def build_graph(
    config: YOLO11Config,
    weights_registry: Mapping[str, object],
    input_shape: tuple[int, int] | None = None,
):
    if input_shape is None:
        height = width = config.input_size
    else:
        height, width = input_shape
    input_type = TensorType(
        config.dtype,
        shape=[1, 3, height, width],
        device=config.device,
    )
    with Graph(f"yolo11{config.size}_detect", input_types=[input_type]) as graph:
        model = YOLO11(config, input_shape=(height, width))
        outputs = model(graph.inputs[0].tensor)
        graph.output(outputs)
    return graph
