from __future__ import annotations

from typing import Any, cast
from pathlib import Path

from max.driver import Device
from max.engine import InferenceSession
from molo.backends.max.pipelines.architectures import get_architecture
from molo.backends.max.pipelines.architectures.yolo.artifact import (
    YOLOArtifact,
    load_artifact,
)
from molo.backends.max.pipelines.architectures.yolo.config import (
    ImageShape,
    YOLOConfig,
    YOLOSize,
)

import numpy as np
import torch

from ultralytics.nn.autobackend import AutoBackend
from ultralytics.nn.backends.base import BaseBackend

MAX_EXTENSION = ".max"


def _has_max_extension(path: str) -> bool:
    return path.lower().endswith(MAX_EXTENSION)


class MaxBackend(BaseBackend):
    def load_model(self, weight):
        weight_path = Path(weight)
        self.weight = weight_path
        self.device = torch.device("cpu")
        self.dynamic = True
        self.task = "detect"
        self.artifact: YOLOArtifact | None = None
        self.config = None
        self.session = None
        self._compiled_models: dict[ImageShape, object] = {}
        self._build_graph: Any = None

        artifact = load_artifact(weight_path)
        self.artifact = artifact
        self.names = dict(enumerate(artifact.class_names))
        model_id = artifact.model_id
        architecture = get_architecture(model_id)
        self._build_graph = architecture.build_graph
        box_name, box_weights = next(
            (name, weight)
            for name, weight in artifact.weights.items()
            if name.endswith("cv2.0.2.weight")
        )
        self.end2end = ".one2one_cv2." in box_name
        self.config = YOLOConfig(
            size=cast(YOLOSize, model_id.removeprefix(architecture.prefix)),
            nc=len(self.names),
            reg_max=box_weights.shape[0] // 4,
        )
        self.fp16 = False
        self.imgsz = self.config.input_size
        self.metadata = {
            "batch": self.batch,
            "channels": self.channels,
            "imgsz": self.imgsz,
            "names": self.names,
            "stride": self.stride,
            "task": self.task,
            "end2end": self.end2end,
        }
        self.session = InferenceSession(devices=[Device.cpu()])
        return self

    def _ensure_model_for_shape(
        self, batch_size: int, input_height: int, input_width: int
    ):
        if (
            self.session is None
            or self.config is None
            or self.artifact is None
            or self._build_graph is None
        ):
            raise RuntimeError("MAX model has not been loaded.")

        shape_key = ImageShape(
            height=int(input_height),
            width=int(input_width),
            batch_size=int(batch_size),
        )
        model = self._compiled_models.get(shape_key)
        if model is None:
            graph = self._build_graph(
                self.config,
                self.artifact,
                input_shape=shape_key,
            )
            model = self.session.load(graph, weights_registry=self.artifact.weights)
            self._compiled_models[shape_key] = model
        self.model = model
        return model

    def forward(self, im: torch.Tensor) -> torch.Tensor:
        assert im.ndim == 4, "Expected BCHW input"
        model = self._ensure_model_for_shape(im.shape[0], im.shape[2], im.shape[3])
        outputs = model.execute(np.ascontiguousarray(im.numpy()))
        assert len(outputs) == 1, "Expected one MAX graph output"
        return torch.from_dlpack(outputs[0])


def install_max_backend() -> None:
    if MAX_EXTENSION in AutoBackend._BACKEND_MAP:
        return

    _model_type_original = AutoBackend._model_type

    def _model_type(self, p: str = "path/to/model.pt", dnn: bool = False) -> str:
        return MAX_EXTENSION if _has_max_extension(p) else _model_type_original(p, dnn)

    AutoBackend._model_type = _model_type  # ty: ignore[invalid-assignment]
    AutoBackend._BACKEND_MAP[MAX_EXTENSION] = MaxBackend  # ty: ignore[invalid-assignment]
