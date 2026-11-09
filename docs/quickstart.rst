.. Copyright (C) 2026 RidgeRun, LLC (http://www.ridgerun.com)
   SPDX-License-Identifier: AGPL-3.0-only

Quickstart
==========

MOLO ports Ultralytics YOLO models to Mojo / MAX. **The project is under
construction.**

``molo`` is the command-line driver: a **drop-in replacement** for the
Ultralytics ``yolo`` command, using the same command syntax and adding MAX
export and inference support. The driver wraps Ultralytics; the ported models
are implemented separately in MOLO's MAX backend.

With Git and `uv <https://docs.astral.sh/uv/getting-started/installation/>`_
installed:

1. Clone and install
--------------------

.. code-block:: bash

   git clone https://github.com/ridgerun-ai/molo.git
   cd molo
   uv sync

2. Convert a model
------------------

.. code-block:: bash

   uv run molo export model=yolo26n.pt format=max

This downloads the checkpoint if needed and creates ``yolo26n.max``.

3. Run inference
----------------

.. code-block:: bash

   uv run molo predict model=yolo26n.max source=https://ultralytics.com/images/bus.jpg device=cpu

.. note::

   The first inference must wait for MAX to compile the model graph. This can
   take several minutes before a prediction appears. Subsequent inferences
   with the same input shape and batch size reuse the graph within the same
   process.

The command prints where it saves the annotated image.
