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

    @exporter.try_export
    def export_max(self, prefix=exporter.colorstr("MAX:")) -> Path:
        from molo.backends.max.pipelines.architectures.yolo.convert import (
            convert_model,
        )

        if self.model.end2end:
            if self.args.max_det != 300:
                raise ValueError("MAX end-to-end export requires max_det=300")
            if self.args.agnostic_nms:
                raise ValueError("MAX end-to-end export requires agnostic_nms=False")
            # Restore the requested limit after Ultralytics' small-image clamp.
            self.model.model[-1].max_det = self.args.max_det

        destination = self.file.with_suffix(".max")
        convert_model(self.model, destination, model_id=Path(self.args.model).stem)
        return destination

    setattr(exporter, "export_formats", export_formats)
    setattr(exporter.Exporter, "export_max", export_max)
    setattr(exporter.Exporter, "_molo_max_export_installed", True)
