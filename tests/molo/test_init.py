# Copyright (C) 2026 RidgeRun, LLC (http://www.ridgerun.com)
# SPDX-License-Identifier: AGPL-3.0-only

"""Tests for the public MOLO API."""

import unittest
from ultralytics import YOLO as UltralyticsYOLO
from molo import YOLO


class PublicAPITest(unittest.TestCase):
    def test_yolo_api(self):
        self.assertIs(YOLO, UltralyticsYOLO)
