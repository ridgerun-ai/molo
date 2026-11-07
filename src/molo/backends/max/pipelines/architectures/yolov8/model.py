"""YOLOv8 detection model for MAX."""

from ultralytics.nn.tasks import DetectionModel
from dataclasses import dataclass
from math import ceil

from ..yolo.config import ImageShape, YOLOConfig, YOLOSize
from ..yolo.artifact import YOLOArtifact
from ..yolo.convert import assign_module

from max import nn
from max.dtype import DType
from max.graph import Graph, TensorType

from ..yolo.layers.conv import Conv
from ..yolo.layers import Concat
from .layers.detect import YOLOv8Detect
from ..yolo.layers.sppf import SPPF
from ..yolo.layers.upsample import UpsampleNearest2x
from .layers.c2f import C2f


@dataclass(frozen=True)
class YOLOv8Scale:
    depth: float
    width: float
    max_channels: int


YOLOV8_SIZES: dict[YOLOSize, YOLOv8Scale] = {
    "n": YOLOv8Scale(depth=0.33, width=0.25, max_channels=1024),
    "s": YOLOv8Scale(depth=0.33, width=0.50, max_channels=1024),
    "m": YOLOv8Scale(depth=0.67, width=0.75, max_channels=768),
    "l": YOLOv8Scale(depth=1.0, width=1.0, max_channels=512),
    "x": YOLOv8Scale(depth=1.0, width=1.25, max_channels=512),
}


class YOLOv8(nn.Module):
    """YOLOv8 detection model, parameterized by size."""

    def __init__(self, config: YOLOConfig) -> None:
        super().__init__()
        self.config = config
        if config.dtype != DType.float32 or config.reg_max != 16:
            raise ValueError("YOLOv8 currently supports float32 and reg_max=16")
        scale = YOLOV8_SIZES[config.size]

        def channels(base: int) -> int:
            if base == config.nc:
                return base
            return ceil(min(base, scale.max_channels) * scale.width / 8) * 8

        c64, c128, c256, c512, c1024 = (channels(c) for c in (64, 128, 256, 512, 1024))
        device = config.device

        def down(ci: int, co: int) -> Conv:
            return Conv(ci, co, kernel_size=3, stride=2, device=device)

        def block(ci: int, co: int, depth: int = 3, shortcut: bool = False) -> C2f:
            return C2f(
                ci,
                co,
                repeats=max(round(depth * scale.depth), 1),
                shortcut=shortcut,
                device=device,
            )

        self.layer0 = down(3, c64)
        self.layer1 = down(c64, c128)
        self.layer2 = block(c128, c128, shortcut=True)
        self.layer3 = down(c128, c256)
        self.layer4 = block(c256, c256, 6, True)
        self.layer5 = down(c256, c512)
        self.layer6 = block(c512, c512, 6, True)
        self.layer7 = down(c512, c1024)
        self.layer8 = block(c1024, c1024, shortcut=True)
        self.layer9 = SPPF(c1024, c1024, kernel_size=5, device=device)
        self.layer10 = UpsampleNearest2x()
        self.layer11 = Concat()
        self.layer12 = block(c1024 + c512, c512)
        self.layer13 = UpsampleNearest2x()
        self.layer14 = Concat()
        self.layer15 = block(c512 + c256, c256)
        self.layer16 = down(c256, c256)
        self.layer17 = Concat()
        self.layer18 = block(c256 + c512, c512)
        self.layer19 = down(c512, c512)
        self.layer20 = Concat()
        self.layer21 = block(c512 + c1024, c1024)
        self.layer22 = YOLOv8Detect(
            num_classes=config.nc,
            in_channels=(c256, c512, c1024),
            device=device,
        )

    def __call__(self, x):
        x = self.layer2(self.layer1(self.layer0(x)))
        x4 = self.layer4(self.layer3(x))
        x6 = self.layer6(self.layer5(x4))
        x9 = self.layer9(self.layer8(self.layer7(x6)))
        x12 = self.layer12(self.layer11([self.layer10(x9), x6]))
        x15 = self.layer15(self.layer14([self.layer13(x12), x4]))
        x18 = self.layer18(self.layer17([self.layer16(x15), x12]))
        x21 = self.layer21(self.layer20([self.layer19(x18), x9]))
        return self.layer22([x15, x18, x21])


def build_graph(
    config: YOLOConfig,
    artifact: YOLOArtifact,
    input_shape: ImageShape | None = None,
) -> Graph:
    shape = input_shape or ImageShape(height=config.input_size, width=config.input_size)
    input_type = TensorType(
        config.dtype, shape=[1, 3, shape.height, shape.width], device=config.device
    )
    with Graph(f"yolov8{config.size}_detect", input_types=[input_type]) as graph:
        model = YOLOv8(config)
        model.load_state_dict(artifact.weights, strict=True)
        graph.output(model(graph.inputs[0].tensor))
    return graph


def convert_model(model: DetectionModel) -> YOLOArtifact:
    """Convert a loaded model to an in-memory MAX artifact."""
    config = identify_model(model)
    weights = {}
    for index in (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 15, 16, 18, 19, 21, 22):
        assign_module(weights, f"layer{index}", model.model[index])
    return YOLOArtifact(
        model_id=f"yolov8{config.size}",
        class_names=[model.names[index] for index in range(config.nc)],
        weights=weights,
    )


def identify_model(model: DetectionModel) -> YOLOConfig:
    """Build a MAX configuration from a loaded Ultralytics model."""
    size = model.yaml.get("scale")
    if not size:
        depth = model.yaml.get("depth_multiple")
        width = model.yaml.get("width_multiple")
        size = next(
            (
                name
                for name, scale in YOLOV8_SIZES.items()
                if (scale.depth, scale.width) == (depth, width)
            ),
            None,
        )
    if size not in YOLOV8_SIZES:
        raise ValueError("Unsupported yolov8 size")
    return YOLOConfig(size=size, nc=model.yaml["nc"], reg_max=model.model[-1].reg_max)
