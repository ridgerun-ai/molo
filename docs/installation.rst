Install and build locally
=========================

Prerequisites
-------------

Install `Git <https://git-scm.com/>`_ and
`uv <https://docs.astral.sh/uv/getting-started/installation/>`_. Use a system
supported by `MAX <https://max.modular.com/system-requirements/>`_. MOLO requires
Python 3.13 or later; uv can install the required interpreter automatically.

Clone and install
-----------------

.. code-block:: bash

   git clone https://github.com/ridgerun-ai/molo.git
   cd molo
   uv sync

This creates a local ``.venv`` and installs MOLO and its dependencies. Run
commands with ``uv run`` from the repository directory. No separate model
compilation step is needed during installation: MAX compiles graphs at runtime.

Build a distribution
--------------------

To build the Python source distribution and wheel:

.. code-block:: bash

   uv build

The distributions are written to ``dist/``. This packages MOLO; it does not
convert checkpoints or compile model graphs.

Continue with :doc:`conversion` to create a MAX artifact.
