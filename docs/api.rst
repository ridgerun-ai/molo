.. Copyright (C) 2026 RidgeRun, LLC (http://www.ridgerun.com)
   SPDX-License-Identifier: AGPL-3.0-only

API and layer reference
=======================

The full package and module reference below includes neural-network layers,
shared YOLO blocks, architecture-specific layers and models, and the
Ultralytics integration.

These pages are generated from the code during each documentation build.

Layers
------

* :doc:`Normalization <generated/molo.backends.max.nn.norm.batch_norm>`
* :doc:`Shared YOLO layers <generated/molo.backends.max.pipelines.architectures.yolo.layers>`
* :doc:`YOLOv8 layers <generated/molo.backends.max.pipelines.architectures.yolov8.layers>`
* :doc:`YOLO11 layers <generated/molo.backends.max.pipelines.architectures.yolo11.layers>`
* :doc:`YOLO26 layers <generated/molo.backends.max.pipelines.architectures.yolo26.layers>`

Models
------

* :doc:`YOLOv8 <generated/molo.backends.max.pipelines.architectures.yolov8.model>`
* :doc:`YOLO11 <generated/molo.backends.max.pipelines.architectures.yolo11.model>`
* :doc:`YOLO26 <generated/molo.backends.max.pipelines.architectures.yolo26.model>`

Package reference
-----------------

.. autosummary::
   :toctree: generated

   molo.backends
   molo.backends.max
   molo.backends.max.nn
   molo.backends.max.nn.norm
   molo.backends.max.nn.norm.batch_norm
   molo.backends.max.pipelines
   molo.backends.max.pipelines.architectures
   molo.backends.max.pipelines.architectures.yolo
   molo.backends.max.pipelines.architectures.yolo.artifact
   molo.backends.max.pipelines.architectures.yolo.config
   molo.backends.max.pipelines.architectures.yolo.convert
   molo.backends.max.pipelines.architectures.yolo.layers
   molo.backends.max.pipelines.architectures.yolo.layers.bottleneck
   molo.backends.max.pipelines.architectures.yolo.layers.concat
   molo.backends.max.pipelines.architectures.yolo.layers.conv
   molo.backends.max.pipelines.architectures.yolo.layers.detect
   molo.backends.max.pipelines.architectures.yolo.layers.dfl
   molo.backends.max.pipelines.architectures.yolo.layers.sppf
   molo.backends.max.pipelines.architectures.yolo.layers.upsample
   molo.backends.max.pipelines.architectures.yolo11
   molo.backends.max.pipelines.architectures.yolo11.layers
   molo.backends.max.pipelines.architectures.yolo11.layers.attention
   molo.backends.max.pipelines.architectures.yolo11.layers.c2psa
   molo.backends.max.pipelines.architectures.yolo11.layers.c3k
   molo.backends.max.pipelines.architectures.yolo11.layers.c3k2
   molo.backends.max.pipelines.architectures.yolo11.layers.detect
   molo.backends.max.pipelines.architectures.yolo11.layers.psa_block
   molo.backends.max.pipelines.architectures.yolo11.model
   molo.backends.max.pipelines.architectures.yolo26
   molo.backends.max.pipelines.architectures.yolo26.layers
   molo.backends.max.pipelines.architectures.yolo26.layers.c3k2_attention
   molo.backends.max.pipelines.architectures.yolo26.layers.detect
   molo.backends.max.pipelines.architectures.yolo26.layers.sppf
   molo.backends.max.pipelines.architectures.yolo26.model
   molo.backends.max.pipelines.architectures.yolov8
   molo.backends.max.pipelines.architectures.yolov8.layers
   molo.backends.max.pipelines.architectures.yolov8.layers.c2f
   molo.backends.max.pipelines.architectures.yolov8.layers.detect
   molo.backends.max.pipelines.architectures.yolov8.model
   molo.models
   molo.models.yolo
   molo.models.yolo.backend
   molo.models.yolo.export
