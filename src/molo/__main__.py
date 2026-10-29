#!/usr/bin/env python3

from ultralytics.cfg import entrypoint

from molo.models.yolo.backend import install_max_backend
from molo.models.yolo.export import install_max_exporter


def main() -> None:
    install_max_backend()
    install_max_exporter()
    entrypoint()


if __name__ == "__main__":
    main()
