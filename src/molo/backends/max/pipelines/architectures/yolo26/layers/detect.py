# Copyright (C) 2026 RidgeRun, LLC (http://www.ridgerun.com)
# SPDX-License-Identifier: AGPL-3.0-only

"""YOLO26 one-to-one detection head."""

from max.graph import DeviceRef, TensorValue, ops
from max.dtype import DType

from ...yolo11.layers.detect import YOLO11Detect


class YOLO26Detect(YOLO11Detect):
    """Decode one-to-one predictions and select detections without NMS.

    Args:
        num_classes: Number of classes.
        in_channels: Channels of the input feature maps.
        strides: Stride of each input feature map.
        device: Target device.
        reg_max: Number of box-distance bins; YOLO26 uses one.

    Returns tensors shaped (batch, detections, 6), containing xyxy coordinates,
    confidence, and class index, matching Ultralytics end-to-end inference.
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: tuple[int, ...],
        strides: tuple[int, ...] = (8, 16, 32),
        device: DeviceRef = DeviceRef.GPU(),
        reg_max: int = 1,
    ) -> None:
        super().__init__(num_classes, in_channels, strides, device, reg_max)
        self.one2one_cv2 = self.cv2
        self.one2one_cv3 = self.cv3
        del self.cv2
        del self.cv3

    def raw(self, features: list[TensorValue]) -> list[TensorValue]:
        """Return raw one-to-one box and classification logits."""
        return self._raw(features, self.one2one_cv2, self.one2one_cv3)

    def _decode_boxes(
        self, top_left: TensorValue, bottom_right: TensorValue
    ) -> TensorValue:
        return ops.concat((top_left, bottom_right), axis=1)

    def postprocess(self, predictions: TensorValue) -> TensorValue:
        """Select the highest-scoring anchor/class pairs independently per image."""
        predictions = ops.permute(predictions, [0, 2, 1])
        batch, anchors, _ = predictions.shape
        boxes, scores = ops.split(predictions, [4, self.num_classes], axis=2)
        k = min(300, int(anchors))
        _, anchor_indices = ops.top_k(ops.squeeze(ops.max(scores, axis=2), axis=2), k)
        selected_scores = ops.gather_nd(
            scores, ops.unsqueeze(anchor_indices, -1), batch_dims=1
        )
        confidence, indices = ops.top_k(ops.reshape(selected_scores, [batch, -1]), k)
        rows = ops.cast(indices // self.num_classes, indices.dtype)
        selected_anchors = ops.gather_nd(
            anchor_indices, ops.unsqueeze(rows, -1), batch_dims=1
        )
        selected_boxes = ops.gather_nd(
            boxes, ops.unsqueeze(selected_anchors, -1), batch_dims=1
        )
        classes = ops.cast(indices % self.num_classes, DType.float32)
        return ops.concat(
            (selected_boxes, ops.unsqueeze(confidence, -1), ops.unsqueeze(classes, -1)),
            axis=2,
        )

    def __call__(self, features: list[TensorValue]) -> TensorValue:
        """Return NMS-free detections from the one-to-one head."""
        return self.postprocess(self.decode(self.raw(features)))
