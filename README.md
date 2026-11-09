# MOLO

MOLO is a port of Ultralytics YOLO detection models to Mojo / MAX. The models
are implemented with MAX's Python graph and neural-network APIs and execute
through the MAX runtime.

To ease testing, MOLO provides `molo`, a drop-in replacement for the Ultralytics
`yolo` command with MAX export and inference support. It also exposes the
Ultralytics-compatible `YOLO` Python API.

**This project is under construction.** The current implementation supports
YOLOv8, YOLO11, and YOLO26 detection models in sizes `n`, `s`, `m`, `l`, and `x`.
Inference on this branch uses CPU and FP32.

## Run locally

Install [Git](https://git-scm.com/) and [uv](https://docs.astral.sh/uv/getting-started/installation/).
Use a system supported by [MAX](https://max.modular.com/system-requirements/).
The project requires Python 3.13 or later; uv can install the required Python
version when setting up the environment.

Clone the repository:

```bash
git clone https://github.com/ridgerun-ai/molo.git
cd molo
uv sync
```

`uv sync` installs the project and its dependencies into a local `.venv`.
No separate compilation or installation of the `molo` command is needed.
Run commands with `uv run` from the repository directory.

## Convert a model

Export an Ultralytics checkpoint to a `.max` artifact:

```bash
uv run molo export model=yolo26n.pt format=max
```

Ultralytics downloads the checkpoint if it is missing. Export creates
`yolo26n.max`, containing the model identifier, class labels, and weights.
The same command works with supported YOLOv8 and YOLO11 checkpoints, for example
`yolov8n.pt` and `yolo11n.pt`.

YOLO26 export requires the default end-to-end configuration, `max_det=300`,
and `agnostic_nms=False`.

## Predict

Run inference on an online image:

```bash
uv run molo predict model=yolo26n.max source=https://ultralytics.com/images/bus.jpg device=cpu
```

For a local image, replace `source` with its path:

```bash
uv run molo predict model=yolo26n.max source=path/to/image.jpg device=cpu
```

The `.max` artifact must already exist. Prediction reads that artifact directly
and does not load the original `.pt` checkpoint. The command reports where it
saves the annotated results.

MAX compiles the graph on the first prediction for each input shape and batch
size. The compiled graph is reused for matching inputs within the same process.

## Python API

```python
from molo import YOLO

model = YOLO("yolo26n.pt")
model.export(format="max")

model = YOLO("yolo26n.max", task="detect")
results = model.predict(
    source="https://ultralytics.com/images/bus.jpg",
    device="cpu",
)
```

## Development

Install development tools and run the tests:

```bash
uv sync --group dev
uv run python -m unittest discover -s tests -t .
```

Some tests download Ultralytics checkpoints when absent. Tests that create
artifacts refuse to overwrite existing files with the same names. Full model
tests compile MAX graphs and can take several minutes.

## Documentation

Build the guides and generated API/layer reference:

```bash
uv sync --group docs
uv run --group docs sphinx-build -W --keep-going -b html docs docs/_build/html
```

Open `docs/_build/html/index.html` in a browser. The documentation includes
installation and package builds, model conversion, inference, development,
and API pages generated from the source docstrings.
