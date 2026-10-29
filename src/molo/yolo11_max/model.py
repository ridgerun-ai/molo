from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from max import nn
from max.dtype import DType
from max.graph import DeviceRef, Graph, TensorType, ops

from .c2psa import C2PSA
from .c3k2 import C3k2
from .conv import Conv
from .detect import Detect
from .sppf import SPPF
from .upsample import UpsampleNearest2x


@dataclass(frozen=True)
class YOLO11Config:
    nc: int = 80
    input_size: int = 640
    reg_max: int = 16
    dtype: DType = DType.float32
    device: DeviceRef = DeviceRef.CPU()


class YOLO11Nano(nn.Module):
    def __init__(self, config: YOLO11Config, input_shape: tuple[int, int]) -> None:
        super().__init__()
        self.config = config
        if config.dtype != DType.float32 or config.reg_max != 16:
            raise ValueError("YOLO11Nano currently supports float32 and reg_max=16")

        self.layer0 = Conv(
            3, 16, kernel_size=3, stride=2, device=config.device, name="layer0"
        )
        self.layer1 = Conv(
            16, 32, kernel_size=3, stride=2, device=config.device, name="layer1"
        )
        self.layer2 = C3k2(
            32,
            64,
            repeats=1,
            c3k=False,
            expansion=0.25,
            device=config.device,
            name="layer2",
        )
        self.layer3 = Conv(
            64, 64, kernel_size=3, stride=2, device=config.device, name="layer3"
        )
        self.layer4 = C3k2(
            64,
            128,
            repeats=1,
            c3k=False,
            expansion=0.25,
            device=config.device,
            name="layer4",
        )
        self.layer5 = Conv(
            128, 128, kernel_size=3, stride=2, device=config.device, name="layer5"
        )
        self.layer6 = C3k2(
            128,
            128,
            repeats=1,
            c3k=True,
            expansion=0.5,
            device=config.device,
            name="layer6",
        )
        self.layer7 = Conv(
            128, 256, kernel_size=3, stride=2, device=config.device, name="layer7"
        )
        self.layer8 = C3k2(
            256,
            256,
            repeats=1,
            c3k=True,
            expansion=0.5,
            device=config.device,
            name="layer8",
        )
        self.layer9 = SPPF(256, 256, kernel_size=5, device=config.device, name="layer9")
        self.layer10 = C2PSA(
            256, 256, repeats=1, expansion=0.5, device=config.device, name="layer10"
        )

        self.upsample0 = UpsampleNearest2x()
        self.layer13 = C3k2(
            256 + 128,
            128,
            repeats=1,
            c3k=False,
            expansion=0.5,
            device=config.device,
            name="layer13",
        )
        self.upsample1 = UpsampleNearest2x()
        self.layer16 = C3k2(
            128 + 128,
            64,
            repeats=1,
            c3k=False,
            expansion=0.5,
            device=config.device,
            name="layer16",
        )
        self.layer17 = Conv(
            64, 64, kernel_size=3, stride=2, device=config.device, name="layer17"
        )
        self.layer19 = C3k2(
            64 + 128,
            128,
            repeats=1,
            c3k=False,
            expansion=0.5,
            device=config.device,
            name="layer19",
        )
        self.layer20 = Conv(
            128, 128, kernel_size=3, stride=2, device=config.device, name="layer20"
        )
        self.layer22 = C3k2(
            128 + 256,
            256,
            repeats=1,
            c3k=True,
            expansion=0.5,
            device=config.device,
            name="layer22",
        )
        self.detect = Detect(
            num_classes=config.nc,
            in_channels=(64, 128, 256),
            device=config.device,
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
    with Graph("yolo11n_detect", input_types=[input_type]) as graph:
        model = YOLO11Nano(config, input_shape=(height, width))
        outputs = model(graph.inputs[0].tensor)
        graph.output(outputs)
    return graph
