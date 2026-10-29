#!/usr/bin/env python3

from ultralytics.cfg import entrypoint

from molo.models.yolo.backend import install_max_backend


def main() -> None:
    install_max_backend()
    entrypoint()


if __name__ == "__main__":
    main()
