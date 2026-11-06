from __future__ import annotations

from ultralytics.nn.tasks import DetectionModel
from dataclasses import dataclass
from math import ceil

from ..yolo.config import ImageShape, YOLOConfig, YOLOSize
from ..yolo.artifact import YOLOArtifact
from ..yolo.convert import assign_module

from max import nn
from max.dtype import DType
from max.graph import Graph, TensorType, ops

from .layers.c2psa import C2PSA
from .layers.c3k2 import C3k2
from ..yolo.layers.conv import Conv
from .layers.detect import YOLO11Detect
from ..yolo.layers.sppf import SPPF
from ..yolo.layers.upsample import UpsampleNearest2x


@dataclass(frozen=True)
class YOLO11Scale:
    """Architecture scaling parameters for a YOLO11 model size."""

    depth: float
    width: float
    max_channels: int
    c3k: bool


YOLO11_SIZES: dict[YOLOSize, YOLO11Scale] = {
    "n": YOLO11Scale(depth=0.5, width=0.25, max_channels=1024, c3k=False),
    "s": YOLO11Scale(depth=0.5, width=0.5, max_channels=1024, c3k=False),
    "m": YOLO11Scale(depth=0.5, width=1.0, max_channels=512, c3k=True),
    "l": YOLO11Scale(depth=1.0, width=1.0, max_channels=512, c3k=True),
    "x": YOLO11Scale(depth=1.0, width=1.5, max_channels=512, c3k=True),
}


class YOLO11(nn.Module):
    """YOLO11 detection model, parameterized by size."""

    def __init__(self, config: YOLOConfig) -> None:
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
        self.detect = YOLO11Detect(
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
    config: YOLOConfig,
    input_shape: ImageShape | None = None,
) -> Graph:
    shape = input_shape or ImageShape(height=config.input_size, width=config.input_size)
    input_type = TensorType(
        config.dtype,
        shape=[1, 3, shape.height, shape.width],
        device=config.device,
    )
    with Graph(f"yolo11{config.size}_detect", input_types=[input_type]) as graph:
        model = YOLO11(config)
        outputs = model(graph.inputs[0].tensor)
        graph.output(outputs)
    return graph


def convert_model(model: DetectionModel) -> YOLOArtifact:
    """Convert a loaded model to an in-memory MAX artifact."""
    config = identify_model(model)
    weights = {}
    for index in (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 16, 17, 19, 20, 22):
        assign_module(weights, f"layer{index}", model.model[index])
    assign_module(weights, "detect", model.model[23])
    return YOLOArtifact(
        model_id=f"yolo11{config.size}",
        class_names=[model.names[index] for index in range(config.nc)],
        weights=weights,
    )


def identify_model(model: DetectionModel) -> YOLOConfig:
    """Build a MAX configuration from a loaded Ultralytics model."""
    size = model.yaml.get("scale")
    if size not in YOLO11_SIZES:
        raise ValueError("Unsupported yolo11 size")
    return YOLOConfig(size=size, nc=model.yaml["nc"], reg_max=model.model[-1].reg_max)
