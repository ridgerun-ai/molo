API and layer reference
=======================

The reference follows the Python package hierarchy. Expand the MAX backend
package for neural-network layers, shared YOLO blocks, and architecture-specific
layers and models. The YOLO integration package documents the Ultralytics
backend and exporter hooks.

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
   :recursive:

   molo.backends
   molo.models
