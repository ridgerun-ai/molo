import ast
from dataclasses import dataclass
from typing import Literal, get_args

import torch
from ultralytics.cfg import check_cfg
from ultralytics.models.yolo.detect import DetectionPredictor
from ultralytics.utils import DEFAULT_CFG_DICT


@dataclass
class Config:
    key: str
    value: object


Task = Literal["segment", "obb", "classify", "pose", "detect"]
Mode = Literal["track", "val", "train", "export", "predict", "benchmark"]

DEFAULT_TASK: Task = "detect"
DEFAULT_ARGS: list[Config] = []


@dataclass
class Arguments:
    mode: Mode
    task: Task
    args: list[Config]


def parse_config(argv: list[str]) -> list[Config]:
    def _parse_token(token: str) -> tuple[str, str]:
        if "=" not in token:
            raise ValueError(f"invalid argument: {token}")

        key, value = token.split("=", 1)
        return key, value

    def _type_pair(pair: tuple[str, str]) -> Config:
        key, value = pair
        try:
            value = ast.literal_eval(value)
        except Exception:
            pass

        return Config(key, value)

    overrides = [_parse_token(a) for a in argv]
    overrides = [_type_pair(o) for o in overrides]

    return overrides


class YOLO11Nano(DetectionPredictor):
    def setup_model(self, model=None, verbose=True):
        self.model = self
        self.device = torch.device("cpu")
        self.stride = 32
        self.fp16 = False
        self.pt = False
        self.triton = False
        self.format = "onnx"
        self.dynamic = False
        self.names = {0: "object"}
        self.channels = 3

    def warmup(self, imgsz=None):
        pass

    def inference(self, im, *args, **kwargs):
        bs = im.shape[0]
        pred = torch.zeros((bs, 1, 6), device=im.device, dtype=torch.float32)
        pred[:, 0, :] = torch.tensor([50, 50, 200, 200, 0.9, 0], device=im.device)
        return pred


def parse_args(argv: list[str]) -> Arguments:
    argc = len(argv)

    tasks = set(get_args(Task))
    modes = set(get_args(Mode))

    match argv:
        # molo
        case [_]:
            raise RuntimeError("MODE is a mandatory argument 1")
        # molo detect y a=b | molo detect y
        case [_, task, mode, *_] if (
            task in tasks and mode not in modes and "=" not in mode
        ):
            raise RuntimeError(f"Invalid value for MODE: {mode} 5")
        # molo x predict a=b | molo x predict
        case [_, task, mode, *_] if task not in tasks and mode in modes:
            raise RuntimeError(f"Invalid value for TASK: {task} 6")
        # molo detect a=b c=d | molo detect a=b | molo a=b c=d | molo c=d
        case [_, task, mode, *_] if task in tasks and "=" in mode:
            raise RuntimeError("MODE is a mandatory argument 4")
        # molo a=b c=d | molo a=b
        case [_, mode, *_] if "=" in mode:
            raise RuntimeError("Mode is a mandatory argument 3")
        # molo x a=b | molo x
        case [_, mode, *_] if mode not in modes and mode not in tasks:
            raise RuntimeError(f"Invalid value for MODE {mode} 3")

        case [_, task, mode, *args] if task in tasks and mode in modes:
            return Arguments(
                task=task,
                mode=mode,
                args=parse_config(args) if argc > 3 else DEFAULT_ARGS,
            )
        case [_, mode, *args] if mode in modes:
            return Arguments(
                task=DEFAULT_TASK,
                mode=mode,
                args=parse_config(args) if argc > 2 else DEFAULT_ARGS,
            )
        case _:
            raise RuntimeError("Invalid invocation")


class App:
    def __call__(self, argv: list[str]):
        arguments = parse_args(argv)

        overrides = {a.key: a.value for a in arguments.args}
        overrides["task"] = arguments.task
        overrides["mode"] = arguments.mode

        check_cfg({**DEFAULT_CFG_DICT, **overrides}, hard=True)

        match arguments:
            case Arguments(task="detect", mode="predict"):
                YOLO11Nano(overrides=overrides).predict_cli()
            case Arguments(task=task, mode=mode, args=_):
                raise RuntimeError(
                    f"Invoking '{task}' with '{mode}' is not supported yet. Only 'detect' with 'predict' is implemented in this version.",
                )
