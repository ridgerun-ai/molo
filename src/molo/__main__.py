#!/usr/bin/env python3

"""Command-line driver for Ultralytics-compatible MAX export and prediction."""

from ultralytics.cfg import entrypoint

from molo.models.yolo import install


def main() -> None:
    install()
    entrypoint()


if __name__ == "__main__":
    main()
