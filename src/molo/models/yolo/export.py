"""Register MAX export with the Ultralytics exporter."""

from pathlib import Path

from ultralytics.engine import exporter


def install_max_exporter() -> None:
    """Add the MAX format and exporter without changing other formats."""
    if getattr(exporter.Exporter, "_molo_max_export_installed", False):
        return

    original_formats = exporter.export_formats

    def export_formats():
        formats = original_formats()
        row = {
            "Format": "MAX",
            "Argument": "max",
            "Suffix": ".max",
            "CPU": True,
            "GPU": False,
            "Arguments": [],
        }
        return {key: (*values, row[key]) for key, values in formats.items()}

    def export_max(self) -> str:
        from molo.backends.max.pipelines.architectures.yolo11.convert import (
            convert_checkpoint,
        )

        source = Path(self.file)
        if source.suffix.lower() != ".pt":
            raise ValueError("MAX export requires a YOLO11n .pt checkpoint")
        destination = source.with_suffix(".max")
        convert_checkpoint(source, destination)
        return str(destination)

    setattr(exporter, "export_formats", export_formats)
    setattr(exporter.Exporter, "export_max", export_max)
    setattr(exporter.Exporter, "_molo_max_export_installed", True)
