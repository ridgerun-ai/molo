# Copyright (C) 2026 RidgeRun, LLC (http://www.ridgerun.com)
# SPDX-License-Identifier: AGPL-3.0-only

"""Tests for the MOLO command entry point."""

import unittest
from unittest.mock import patch
from molo.__main__ import main


class MainTest(unittest.TestCase):
    def test_main(self):
        with (
            patch("molo.__main__.install") as install,
            patch("molo.__main__.entrypoint") as entrypoint,
        ):
            main()
        install.assert_called_once_with()
        entrypoint.assert_called_once_with()
