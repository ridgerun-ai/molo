from __future__ import annotations

from ultralytics.nn.autobackend import AutoBackend
from ultralytics.nn.backends.base import BaseBackend

MAX_EXTENSION = ".max"


def _has_max_extension(path: str) -> bool:
    return path.lower().endswith(MAX_EXTENSION)


class MaxBackend(BaseBackend):
    def load_model(self, weight):
        from pathlib import Path

        import torch

        weight_path = Path(weight)
        self.weight = weight_path
        self.device = torch.device("cpu")
        self.fp16 = False
        self.dynamic = True
        self.nhwc = False
        self.stride = 32
        self.task = "detect"
        self.weights_registry = {}
        self.config = None
        self.session = None
        self.model = None
        self._compiled_models: dict[tuple[int, int], object] = {}
        self._build_graph = None

        try:
            from max.driver import Device
            from max.engine import InferenceSession
            from molo.backends.max.pipelines.architectures.yolo11.artifact import (
                load_artifact,
            )
            from molo.backends.max.pipelines.architectures.yolo11.model import (
                YOLO11Config,
                build_graph,
            )
        except ImportError as exc:
            raise ImportError(
                "The MAX runtime is required to load .max models."
            ) from exc

        self.weights_registry, self.names = load_artifact(weight_path)
        self._build_graph = build_graph
        self.config = YOLO11Config(nc=len(self.names))
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
        if self.session is None or self.config is None or self._build_graph is None:
            raise RuntimeError("MAX model has not been loaded.")

        shape_key = (int(input_height), int(input_width))
        model = self._compiled_models.get(shape_key)
        if model is None:
            graph = self._build_graph(
                self.config,
                self.weights_registry,
                input_shape=shape_key,
            )
            model = self.session.load(graph, weights_registry=self.weights_registry)
            self._compiled_models[shape_key] = model
        self.model = model
        return model

    def forward(self, im):
        import numpy as np
        import torch

        if isinstance(im, torch.Tensor):
            im = im.detach().cpu().numpy()

        im = np.asarray(im)
        if im.ndim == 3:
            im = np.expand_dims(im, axis=0)
        if im.ndim != 4:
            raise ValueError(
                f"Expected BCHW input for MAX backend, got shape {im.shape}."
            )
        if im.shape[0] != 1:
            raise ValueError("MAX backend currently supports batch size 1 only.")
        input_h, input_w = im.shape[2], im.shape[3]
        model = self._ensure_model_for_shape(input_h, input_w)

        im = np.ascontiguousarray(
            im, dtype=np.float16 if getattr(self, "fp16", False) else np.float32
        )
        outputs = model.execute(im)
        if len(outputs) != 1:
            raise ValueError("MAX graph returned an unexpected number of outputs.")

        output = outputs[0]
        if hasattr(output, "__dlpack__"):
            output = np.from_dlpack(output)
        else:
            output = np.asarray(output)
        return torch.from_numpy(output).to(self.device)


def install_max_backend() -> None:
    if MAX_EXTENSION in AutoBackend._BACKEND_MAP:
        return

    _model_type_original = AutoBackend._model_type

    def _model_type(self, p: str = "path/to/model.pt", dnn: bool = False) -> str:
        return MAX_EXTENSION if _has_max_extension(p) else _model_type_original(p, dnn)

    AutoBackend._model_type = _model_type  # ty: ignore[invalid-assignment]
    AutoBackend._BACKEND_MAP[MAX_EXTENSION] = MaxBackend  # ty: ignore[invalid-assignment]
