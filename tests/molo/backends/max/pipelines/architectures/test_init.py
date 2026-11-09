# Copyright (C) 2026 RidgeRun, LLC (http://www.ridgerun.com)
# SPDX-License-Identifier: AGPL-3.0-only

"""Tests for YOLO architecture registration and construction."""

import unittest
from unittest.mock import Mock, patch

from molo.backends.max.pipelines import architectures


class FactoryTest(unittest.TestCase):
    def test_registration(self):
        architecture = architectures.Architecture(
            "another_", {"small", "large"}, Mock(), Mock()
        )
        with patch.object(architectures, "_ARCHITECTURES", {}):
            architectures.register_architecture(architecture)
            self.assertIs(architectures.get_architecture("another_small"), architecture)
            self.assertIs(architectures.get_architecture("another_large"), architecture)

    def test_unknown_identifier(self):
        with self.assertRaisesRegex(ValueError, "Unsupported model identifier"):
            architectures.get_architecture("unknown")
