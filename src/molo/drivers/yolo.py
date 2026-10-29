from __future__ import annotations

from ultralytics.nn.autobackend import AutoBackend
from ultralytics.nn.backends.base import BaseBackend

MAX_EXTENSION = ".max"


def _has_max_extension(path: str) -> bool:
    return path.lower().endswith(MAX_EXTENSION)


class MaxBackend(BaseBackend):
    def load_model(self, weight):
        from pathlib import Path

        import numpy as np
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
            from ultralytics import YOLO
            from molo.yolo11_max import convert as max_convert
            from molo.yolo11_max.model import YOLO11Config, build_graph
        except ImportError as exc:
            raise ImportError(
                "The MAX runtime is required to load .max models."
            ) from exc

        source_path = weight_path
        if source_path.suffix == ".max":
            source_path = source_path.with_suffix(".pt")
        if source_path.suffix != ".pt":
            raise ValueError(
                f"Expected a .max or .pt weight path, got '{source_path}'."
            )
        if not source_path.exists():
            raise FileNotFoundError(
                f"Could not find source checkpoint '{source_path}'."
            )

        yolo = YOLO(str(source_path))
        pt = yolo.model.model  # ty: ignore[unresolved-attribute]
        names = getattr(yolo.model, "names", None) or getattr(yolo, "names", None) or {}
        self.names = dict(names)
        self._build_graph = build_graph
        target: dict[str, np.ndarray] = {}

        for idx in (0, 1, 3, 5, 7, 17, 20):
            max_convert._assign_conv(
                target, f"layer{idx}", pt[idx], max_name=f"layer{idx}"
            )

        for idx in (2, 4, 6, 8, 13, 16, 19, 22):
            max_convert._assign_c3k2(target, f"layer{idx}", pt[idx])

        max_convert._assign_sppf(target, "layer9", pt[9])
        max_convert._assign_c2psa(target, "layer10", pt[10])
        max_convert._assign_detect(target, "detect", pt[23])

        self.weights_registry = target
        self.config = YOLO11Config()
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
