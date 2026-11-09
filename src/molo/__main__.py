#!/usr/bin/env python3

# Copyright (C) 2026 RidgeRun, LLC (http://www.ridgerun.com)
# SPDX-License-Identifier: AGPL-3.0-only

"""Command-line driver for Ultralytics-compatible MAX export and prediction."""

from ultralytics.cfg import entrypoint

from molo.models.yolo import install


def main() -> None:
    install()
    entrypoint()


if __name__ == "__main__":
    main()
