# Copyright (C) 2026 RidgeRun, LLC (http://www.ridgerun.com)
# SPDX-License-Identifier: AGPL-3.0-only

"""YOLOv8 detection head."""

from ...yolo.layers.detect import Detect


class YOLOv8Detect(Detect):
    """Detection head with the YOLOv8 classification branch."""

    def _classification_branch(
        self, channels, class_width, num_classes, conv, output_conv
    ):
        return [
            conv(channels, class_width, 3),
            conv(class_width, class_width, 3),
            output_conv(class_width, num_classes),
        ]
