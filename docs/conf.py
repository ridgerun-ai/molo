# Copyright (C) 2026 RidgeRun, LLC (http://www.ridgerun.com)
# SPDX-License-Identifier: AGPL-3.0-only

"""Sphinx configuration for MOLO's guides and generated API reference."""

from importlib.metadata import version as package_version

project = "MOLO"
author = "RidgeRun, LLC"
copyright = "2026, RidgeRun, LLC"
release = package_version("molo")
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]
templates_path = ["_templates"]
exclude_patterns = ["_build", "_templates"]
autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "member-order": "bysource",
    "special-members": "__call__",
}
autodoc_typehints = "description"
html_theme = "alabaster"
html_title = "MOLO documentation"
html_theme_options = {"github_user": "ridgerun-ai", "github_repo": "molo"}
