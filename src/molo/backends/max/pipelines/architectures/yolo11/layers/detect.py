"""YOLO11 detection head."""

from ...yolo.layers.detect import Detect


class YOLO11Detect(Detect):
    """Detection head with the YOLO11 classification branch."""

    def _classification_branch(
        self, channels, class_width, num_classes, index, conv, output_conv
    ):
        return [
            conv(channels, channels, 3, f"cv3.{index}.0.0", groups=channels),
            conv(channels, class_width, 1, f"cv3.{index}.0.1"),
            conv(class_width, class_width, 3, f"cv3.{index}.1.0", groups=class_width),
            conv(class_width, class_width, 1, f"cv3.{index}.1.1"),
            output_conv(class_width, num_classes, f"cv3.{index}.2"),
        ]
