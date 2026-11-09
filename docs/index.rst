MOLO
====

MOLO is a port of Ultralytics YOLO detection models to Mojo / MAX. Models use
MAX's Python graph and neural-network APIs and run through the MAX runtime.

To ease testing, MOLO provides ``molo``, a drop-in replacement driver for the
Ultralytics ``yolo`` command with MAX export and prediction support. The Python
interface also exposes the Ultralytics-compatible ``YOLO`` API.

.. warning::

   This project is under construction. This branch supports CPU inference with
   FP32 weights for YOLOv8, YOLO11, and YOLO26 detection models, in sizes
   ``n``, ``s``, ``m``, ``l``, and ``x``.

.. toctree::
   :maxdepth: 2

   quickstart
   installation
   conversion
   inference
   development
   api

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
