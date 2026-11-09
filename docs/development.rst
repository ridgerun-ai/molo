.. Copyright (C) 2026 RidgeRun, LLC (http://www.ridgerun.com)
   SPDX-License-Identifier: AGPL-3.0-only

Development and documentation
=============================

Tests and checks
----------------

.. code-block:: bash

   uv sync --group dev
   uv run python -m unittest discover -s tests -t .
   uv run ruff check src tests
   uv run ty check src tests

Tests mirror the source package layout under ``tests/molo``. Layer tests
compare against the corresponding Ultralytics or PyTorch implementation across
configurations. Full model tests compile MAX graphs and can take several
minutes. Checkpoints are downloaded when absent; artifact-writing tests refuse
to overwrite existing files with the same names.

Build the documentation
-----------------------

.. code-block:: bash

   uv sync --group docs
   uv run --group docs sphinx-build -W --keep-going -b html docs docs/_build/html

Open ``docs/_build/html/index.html`` in a browser, or serve it locally:

.. code-block:: bash

   uv run python -m http.server 8000 --directory docs/_build/html

Visit http://localhost:8000. Stop the server with ``Ctrl+C``.

The build uses Sphinx autodoc and recursive autosummary to generate the API
reference from source signatures and docstrings. Napoleon renders Google-style
``Args``, ``Returns``, and ``Raises`` sections. Update the source docstrings when
changing a layer's API. Generated reference pages and HTML output are ignored
by Git and recreated during the build.
