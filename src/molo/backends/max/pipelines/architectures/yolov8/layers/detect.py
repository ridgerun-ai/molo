"""YOLOv8 detection head."""

from ...yolo.layers.detect import Detect


class YOLOv8Detect(Detect):
    """Detection head with the YOLOv8 classification branch."""

    def _classification_branch(
        self, channels, class_width, num_classes, index, conv, output_conv
    ):
        return [
            conv(channels, class_width, 3, f"cv3.{index}.0"),
            conv(class_width, class_width, 3, f"cv3.{index}.1"),
            output_conv(class_width, num_classes, f"cv3.{index}.2"),
        ]
