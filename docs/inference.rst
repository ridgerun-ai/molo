Run inference
=============

Command line
------------

First create the artifact as described in :doc:`conversion`. Then run:

.. code-block:: bash

   uv run molo predict model=yolo26n.max source=https://ultralytics.com/images/bus.jpg device=cpu

For a local image:

.. code-block:: bash

   uv run molo predict model=yolo26n.max source=path/to/image.jpg device=cpu

The command prints the output directory containing annotated predictions.
The artifact must already exist. ``molo predict`` does not convert a missing
artifact or load a source checkpoint in its place.

Use the usual Ultralytics prediction arguments, for example:

.. code-block:: bash

   uv run molo predict model=yolo11n.max source=path/to/images device=cpu imgsz=640 conf=0.25 batch=2

This branch supports CPU and FP32 execution. GPU support is separate work.

Python API
----------

MOLO exposes the Ultralytics ``YOLO`` class after installing its MAX integration:

.. code-block:: python

   from molo import YOLO

   model = YOLO("yolo26n.max", task="detect")
   results = model.predict(
       source="https://ultralytics.com/images/bus.jpg",
       device="cpu",
   )
   for result in results:
       print(result.boxes)

Results use the normal Ultralytics result objects. Preprocessing, annotation,
and postprocessing use the Ultralytics prediction pipeline; the neural network
executes through MAX.

Compilation and batch sizes
---------------------------

The backend compiles a graph on the first forward pass for each batch size,
height, and width. Matching inputs reuse the compiled graph within that backend
instance. A new process starts with an empty graph cache, so its first
prediction includes compilation time.

YOLOv8 and YOLO11 produce predictions before non-maximum suppression (NMS),
which Ultralytics applies. YOLO26 uses its end-to-end detection head and returns
selected detections without NMS.
