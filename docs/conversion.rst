Convert models
==============

Command line
------------

Convert an Ultralytics checkpoint using the normal export syntax:

.. code-block:: bash

   uv run molo export model=yolo26n.pt format=max

Ultralytics downloads a supported checkpoint if it is absent. This creates
``yolo26n.max`` next to the source checkpoint. Supported model identifiers are:

.. list-table::
   :header-rows: 1

   * - Family
     - Checkpoint names
   * - YOLOv8
     - ``yolov8n.pt``, ``yolov8s.pt``, ``yolov8m.pt``, ``yolov8l.pt``, ``yolov8x.pt``
   * - YOLO11
     - ``yolo11n.pt``, ``yolo11s.pt``, ``yolo11m.pt``, ``yolo11l.pt``, ``yolo11x.pt``
   * - YOLO26
     - ``yolo26n.pt``, ``yolo26s.pt``, ``yolo26m.pt``, ``yolo26l.pt``, ``yolo26x.pt``

The current implementation uses FP32 weights. YOLO26 export requires
``end2end=True``, ``max_det=300``, and ``agnostic_nms=False``, the default
configuration. Unsupported YOLO26 selection settings are rejected.

Python API
----------

.. code-block:: python

   from molo import YOLO

   model = YOLO("yolo26n.pt")
   artifact_path = model.export(format="max")

What is exported?
-----------------

The ``.max`` file is a self-contained NumPy archive containing model weights,
the model identifier, and class labels. It is not a precompiled executable.
Prediction loads this artifact directly, without loading the original ``.pt``
checkpoint.

Ultralytics prepares and fuses the model before invoking the MAX converter.
The architecture converter returns an in-memory artifact, and the shared
export path writes it to disk. Model construction uses strict MAX weight
loading. Runtime graph compilation is described in :doc:`inference`.
