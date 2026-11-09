# Copyright (C) 2026 RidgeRun, LLC (http://www.ridgerun.com)
# SPDX-License-Identifier: AGPL-3.0-only

"""Register MOLO's Ultralytics integrations."""

from .backend import install_max_backend
from .export import install_max_exporter


def install() -> None:
    """Register MAX inference and export with Ultralytics."""
    install_max_backend()
    install_max_exporter()
