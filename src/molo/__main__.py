#!/usr/bin/env python3

from ultralytics.cfg import entrypoint

from molo.models.yolo import install


def main() -> None:
    install()
    entrypoint()


if __name__ == "__main__":
    main()
