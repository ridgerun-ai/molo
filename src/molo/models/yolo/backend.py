from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast
from pathlib import Path

import numpy as np
import torch

if TYPE_CHECKING:
    from molo.backends.max.pipelines.architectures.yolo.config import ImageShape
    from molo.backends.max.pipelines.architectures.yolo.artifact import YOLOArtifact

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

        try:
            from max.dtype import DType
            from max.driver import Device
            from max.engine import InferenceSession
            from molo.backends.max.pipelines.architectures.yolo.artifact import (
                load_artifact,
            )
            from molo.backends.max.pipelines.architectures import get_architecture
            from molo.backends.max.pipelines.architectures.yolo.config import (
                YOLOConfig,
                YOLOSize,
            )
        except ImportError as exc:
            raise ImportError(
                "The MAX runtime is required to load .max models."
            ) from exc

        artifact = load_artifact(weight_path)
        self.artifact = artifact
        self.names = dict(enumerate(artifact.class_names))
        model_id = artifact.model_id
        architecture = get_architecture(model_id)
        self._build_graph = architecture.build_graph
        self.config = YOLOConfig(
            size=cast(YOLOSize, model_id.removeprefix(architecture.prefix)),
            nc=len(self.names),
        )
        self.fp16 = self.config.dtype == DType.float16
        self.imgsz = self.config.input_size
        self.metadata = {
            "batch": self.batch,
            "channels": self.channels,
            "imgsz": self.imgsz,
            "names": self.names,
            "stride": self.stride,
            "task": self.task,
        }
        self.session = InferenceSession(devices=[Device.cpu()])
        return self

    def _ensure_model_for_shape(self, input_height: int, input_width: int):
        from molo.backends.max.pipelines.architectures.yolo.config import ImageShape

        if (
            self.session is None
            or self.config is None
            or self.artifact is None
            or self._build_graph is None
        ):
            raise RuntimeError("MAX model has not been loaded.")

        shape_key = ImageShape(height=int(input_height), width=int(input_width))
        model = self._compiled_models.get(shape_key)
        if model is None:
            graph = self._build_graph(
                self.config,
                input_shape=shape_key,
            )
            model = self.session.load(graph, weights_registry=self.artifact.weights)
            self._compiled_models[shape_key] = model
        self.model = model
        return model

    def forward(self, im: torch.Tensor) -> torch.Tensor:
        assert im.ndim == 4, "Expected BCHW input"
        assert im.shape[0] == 1, "MAX backend supports batch size 1 only"
        model = self._ensure_model_for_shape(im.shape[2], im.shape[3])
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
