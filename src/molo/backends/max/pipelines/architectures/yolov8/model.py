# Copyright (C) 2026 RidgeRun, LLC (http://www.ridgerun.com)
# SPDX-License-Identifier: AGPL-3.0-only

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
from ..yolo.layers.concat import Concat
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

        self.model = nn.LayerList(
            [
                down(3, c64),
                down(c64, c128),
                block(c128, c128, shortcut=True),
                down(c128, c256),
                block(c256, c256, 6, True),
                down(c256, c512),
                block(c512, c512, 6, True),
                down(c512, c1024),
                block(c1024, c1024, shortcut=True),
                SPPF(c1024, c1024, kernel_size=5, device=device),
                UpsampleNearest2x(),
                Concat(),
                block(c1024 + c512, c512),
                UpsampleNearest2x(),
                Concat(),
                block(c512 + c256, c256),
                down(c256, c256),
                Concat(),
                block(c256 + c512, c512),
                down(c512, c512),
                Concat(),
                block(c512 + c1024, c1024),
                YOLOv8Detect(
                    num_classes=config.nc,
                    reg_max=config.reg_max,
                    in_channels=(c256, c512, c1024),
                    device=device,
                ),
            ]
        )

    def __call__(self, x):
        x = self.model[2](self.model[1](self.model[0](x)))
        x4 = self.model[4](self.model[3](x))
        x6 = self.model[6](self.model[5](x4))
        x9 = self.model[9](self.model[8](self.model[7](x6)))
        x12 = self.model[12](self.model[11]([self.model[10](x9), x6]))
        x15 = self.model[15](self.model[14]([self.model[13](x12), x4]))
        x18 = self.model[18](self.model[17]([self.model[16](x15), x12]))
        x21 = self.model[21](self.model[20]([self.model[19](x18), x9]))
        return self.model[22]([x15, x18, x21])


def build_graph(
    config: YOLOConfig,
    artifact: YOLOArtifact,
    input_shape: ImageShape | None = None,
) -> Graph:
    shape = input_shape or ImageShape(height=config.input_size, width=config.input_size)
    input_type = TensorType(
        DType.float32,
        shape=[shape.batch_size, 3, shape.height, shape.width],
        device=config.device,
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
    assign_module(weights, "model", model.model)
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
