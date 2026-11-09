# Copyright (C) 2026 RidgeRun, LLC (http://www.ridgerun.com)
# SPDX-License-Identifier: AGPL-3.0-only

"""YOLO11 model construction, graph building, and weight conversion."""

from __future__ import annotations

from ultralytics.nn.tasks import DetectionModel
from dataclasses import dataclass
from math import ceil

from ..yolo.config import ImageShape, YOLOConfig, YOLOSize
from ..yolo.artifact import YOLOArtifact
from ..yolo.convert import assign_module

from max import nn
from max.dtype import DType
from max.graph import Graph, TensorType

from .layers.c2psa import C2PSA
from .layers.c3k2 import C3k2
from ..yolo.layers.conv import Conv
from ..yolo.layers.concat import Concat
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

        def channels(base: int) -> int:
            if base == config.nc:
                return base
            return ceil(min(base, scale.max_channels) * scale.width / 8) * 8

        repeats = max(round(2 * scale.depth), 1)
        c64, c128, c256, c512, c1024 = (channels(c) for c in (64, 128, 256, 512, 1024))
        device = config.device

        def down(ci: int, co: int) -> Conv:
            return Conv(ci, co, kernel_size=3, stride=2, device=device)

        def block(
            ci: int, co: int, *, c3k: bool = False, expansion: float = 0.5
        ) -> C3k2:
            return C3k2(
                ci,
                co,
                repeats=repeats,
                c3k=c3k or scale.c3k,
                expansion=expansion,
                device=device,
            )

        self.model = nn.LayerList(
            [
                down(3, c64),
                down(c64, c128),
                block(c128, c256, expansion=0.25),
                down(c256, c256),
                block(c256, c512, expansion=0.25),
                down(c512, c512),
                block(c512, c512, c3k=True),
                down(c512, c1024),
                block(c1024, c1024, c3k=True),
                SPPF(c1024, c1024, kernel_size=5, device=device),
                C2PSA(c1024, c1024, repeats=repeats, expansion=0.5, device=device),
                UpsampleNearest2x(),
                Concat(),
                block(c1024 + c512, c512),
                UpsampleNearest2x(),
                Concat(),
                block(c512 + c512, c256),
                down(c256, c256),
                Concat(),
                block(c256 + c512, c512),
                down(c512, c512),
                Concat(),
                block(c512 + c1024, c1024, c3k=True),
                YOLO11Detect(
                    num_classes=config.nc,
                    reg_max=config.reg_max,
                    in_channels=(c256, c512, c1024),
                    device=device,
                ),
            ]
        )

    def __call__(self, x):
        x0 = self.model[0](x)
        x1 = self.model[1](x0)
        x2 = self.model[2](x1)
        x3 = self.model[3](x2)
        x4 = self.model[4](x3)
        x5 = self.model[5](x4)
        x6 = self.model[6](x5)
        x7 = self.model[7](x6)
        x8 = self.model[8](x7)
        x9 = self.model[9](x8)
        x10 = self.model[10](x9)

        x11 = self.model[11](x10)
        x12 = self.model[13](self.model[12]([x11, x6]))
        x13 = self.model[14](x12)
        x14 = self.model[16](self.model[15]([x13, x4]))
        x15 = self.model[17](x14)
        x16 = self.model[19](self.model[18]([x15, x12]))
        x17 = self.model[20](x16)
        x18 = self.model[22](self.model[21]([x17, x10]))
        return self.model[23]([x14, x16, x18])


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
    with Graph(f"yolo11{config.size}_detect", input_types=[input_type]) as graph:
        model = YOLO11(config)
        model.load_state_dict(artifact.weights, strict=True)
        outputs = model(graph.inputs[0].tensor)
        graph.output(outputs)
    return graph


def convert_model(model: DetectionModel) -> YOLOArtifact:
    """Convert a loaded model to an in-memory MAX artifact."""
    config = identify_model(model)
    weights = {}
    assign_module(weights, "model", model.model)
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
