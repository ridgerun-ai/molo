"""Shared YOLO object-detection head for MAX Graph inference."""

import numpy as np
from max.dtype import DType
from max.graph import DeviceRef, TensorValue, ops
from max.nn import Conv2d, Module
from .conv import Conv
from .dfl import DFL


class Detect(Module):
    """Shared detection head with separate box and classification branches.

    Args:
        num_classes: Number of classes.
        in_channels: Input channels at each feature-map scale.
        strides: Image stride for each feature-map scale.
        device: Target device.
        name: Optional weight-name prefix.

    The input feature maps use NCHW layout. Inference returns decoded boxes in
    xywh format and sigmoid class scores, shaped (batch, 4 + classes, locations).
    Feature-map spatial dimensions must be static when the graph is built.
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: tuple[int, ...],
        strides: tuple[int, ...] = (8, 16, 32),
        device: DeviceRef = DeviceRef.GPU(),
        name: str | None = None,
    ) -> None:
        super().__init__()
        if num_classes <= 0 or not in_channels or len(in_channels) != len(strides):
            raise ValueError(
                "provide positive class count and matching channels and strides"
            )
        self.num_classes = num_classes
        self.in_channels = tuple(in_channels)
        self.strides = tuple(strides)
        self.bins = 16
        self.dfl = DFL(self.bins)
        prefix = f"{name}." if name else ""
        box_width = max(16, in_channels[0] // 4, 4 * self.bins)
        class_width = max(in_channels[0], min(num_classes, 100))
        box_branches = []
        class_branches = []
        for i, channels in enumerate(in_channels):

            def conv(ci, co, k, weight_name, groups=1):
                return Conv(
                    ci,
                    co,
                    kernel_size=k,
                    groups=groups,
                    device=device,
                    name=f"{prefix}{weight_name}",
                )

            def output_conv(ci, co, weight_name):
                return Conv2d(
                    1,
                    ci,
                    co,
                    DType.float32,
                    device=device,
                    has_bias=True,
                    permute=True,
                    name=f"{prefix}{weight_name}",
                )

            box_branches.append(
                [
                    conv(channels, box_width, 3, f"cv2.{i}.0"),
                    conv(box_width, box_width, 3, f"cv2.{i}.1"),
                    output_conv(box_width, 4 * self.bins, f"cv2.{i}.2"),
                ]
            )
            class_branches.append(
                self._classification_branch(
                    channels, class_width, num_classes, i, conv, output_conv
                )
            )

        self.box_branches = box_branches
        self.class_branches = class_branches

    def _classification_branch(
        self, channels, class_width, num_classes, index, conv, output_conv
    ):
        raise NotImplementedError

    def raw(self, features: list[TensorValue]) -> list[TensorValue]:
        """Return the unprocessed box and class logits for each scale."""
        if len(features) != len(self.in_channels):
            raise ValueError("number of feature maps must match in_channels")
        outputs = []
        for feature, box_branch, class_branch in zip(
            features, self.box_branches, self.class_branches
        ):
            box = feature
            for layer in box_branch:
                box = layer(box)
            cls = feature
            for layer in class_branch:
                cls = layer(cls)
            outputs.append(ops.concat((box, cls), axis=1))
        return outputs

    def decode(self, raw: list[TensorValue]) -> TensorValue:
        """Decode raw logits into xywh boxes and class scores."""
        if len(raw) != len(self.strides):
            raise ValueError("number of raw feature maps must match strides")
        flattened, anchors, strides = [], [], []
        for feature, stride in zip(raw, self.strides):
            batch, _, h, w = feature.shape
            try:
                height, width = int(h), int(w)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "Detect requires static feature-map heights and widths"
                ) from exc
            flattened.append(
                ops.reshape(feature, (batch, 4 * self.bins + self.num_classes, -1))
            )
            yy, xx = np.meshgrid(
                np.arange(height, dtype=np.float32) + 0.5,
                np.arange(width, dtype=np.float32) + 0.5,
                indexing="ij",
            )
            anchors.append(np.stack((xx.ravel(), yy.ravel())))
            strides.append(np.full((1, height * width), stride, dtype=np.float32))
        combined = ops.concat(flattened, axis=2)
        boxes, classes = ops.split(combined, [4 * self.bins, self.num_classes], axis=1)
        distances = self.dfl(boxes)
        left_top, right_bottom = ops.split(distances, [2, 2], axis=1)
        points = ops.constant(
            np.concatenate(anchors, axis=1)[None],
            dtype=DType.float32,
            device=DeviceRef.CPU(),
        ).to(combined.device)
        scale = ops.constant(
            np.concatenate(strides, axis=1)[None],
            dtype=DType.float32,
            device=DeviceRef.CPU(),
        ).to(combined.device)
        top_left, bottom_right = points - left_top, points + right_bottom
        center = (top_left + bottom_right) * 0.5
        size = bottom_right - top_left
        return ops.concat(
            (ops.concat((center, size), axis=1) * scale, ops.sigmoid(classes)), axis=1
        )

    def __call__(self, features: list[TensorValue]) -> TensorValue:
        """Return decoded boxes and class scores before NMS."""
        return self.decode(self.raw(features))
