"""Tests for the public MOLO API."""

import unittest
from ultralytics import YOLO as UltralyticsYOLO
from molo import YOLO


class PublicAPITest(unittest.TestCase):
    def test_yolo_api(self):
        self.assertIs(YOLO, UltralyticsYOLO)
