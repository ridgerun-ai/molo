"""Tests for YOLO integration registration."""

import unittest
from unittest.mock import patch
from molo.models.yolo import install


class InstallTest(unittest.TestCase):
    def test_install(self):
        with (
            patch("molo.models.yolo.install_max_backend") as backend,
            patch("molo.models.yolo.install_max_exporter") as exporter,
        ):
            install()
        backend.assert_called_once_with()
        exporter.assert_called_once_with()
