from __future__ import annotations

from ultralytics.nn.autobackend import AutoBackend
from ultralytics.nn.backends.base import BaseBackend

MAX_EXTENSION = ".max"


def _has_max_extension(path: str) -> bool:
    return path.lower().endswith(MAX_EXTENSION)


class MaxBackend(BaseBackend):
    def load_model(self, weight):
        pass

    def forward(self, im):
        raise NotImplementedError


def install_max_backend() -> None:
    if MAX_EXTENSION in AutoBackend._BACKEND_MAP:
        return

    _model_type_original = AutoBackend._model_type

    def _model_type(self, p: str = "path/to/model.pt", dnn: bool = False) -> str:
        return MAX_EXTENSION if _has_max_extension(p) else _model_type_original(p, dnn)

    AutoBackend._model_type = _model_type  # ty: ignore[invalid-assignment]
    AutoBackend._BACKEND_MAP[MAX_EXTENSION] = MaxBackend  # ty: ignore[invalid-assignment]
