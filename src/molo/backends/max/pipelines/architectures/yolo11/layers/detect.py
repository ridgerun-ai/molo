"""YOLO11 detection head."""

from max.nn import Sequential

from ...yolo.layers.detect import Detect


class YOLO11Detect(Detect):
    """Detection head with the YOLO11 classification branch."""

    def _classification_branch(
        self, channels, class_width, num_classes, conv, output_conv
    ):
        return [
            Sequential(
                [
                    conv(channels, channels, 3, groups=channels),
                    conv(channels, class_width, 1),
                ]
            ),
            Sequential(
                [
                    conv(
                        class_width,
                        class_width,
                        3,
                        groups=class_width,
                    ),
                    conv(class_width, class_width, 1),
                ]
            ),
            output_conv(class_width, num_classes),
        ]
