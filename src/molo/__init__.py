"""Ultralytics-compatible Python interface with MAX support."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ultralytics import YOLO as YOLO

__all__ = ["YOLO"]


def __getattr__(name: str):
    """Expose Ultralytics YOLO after registering MOLO integrations."""
    if name != "YOLO":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from ultralytics import YOLO

    from molo.models.yolo import install

    install()
    return YOLO
