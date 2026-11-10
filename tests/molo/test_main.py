# Copyright (C) 2026 RidgeRun, LLC (http://www.ridgerun.com)
# SPDX-License-Identifier: AGPL-3.0-only

"""Tests for the MOLO command entry point."""

import unittest
from unittest.mock import patch
from molo.__main__ import main
from ultralytics.cfg import merge_equals_args, parse_key_value_pair


class MainTest(unittest.TestCase):
    def test_main(self):
        with (
            patch("molo.__main__.ARGV", ["molo", "predict"]),
            patch("molo.__main__.install") as install,
            patch("molo.__main__.entrypoint") as entrypoint,
        ):
            main()
        install.assert_called_once_with()
        entrypoint.assert_called_once_with()

    def test_prediction_save_defaults(self):
        for mode in ("predict", "track", "mode=predict", "mode=track"):
            for options, expected in (
                ([], True),
                (["save=False"], False),
                (["save=True"], True),
            ):
                arguments = ["molo", mode, "source=image.jpg", *options]
                with self.subTest(mode=mode, options=options):

                    def check_arguments():
                        parsed = dict(
                            parse_key_value_pair(arg)
                            for arg in merge_equals_args(arguments[1:])
                            if "=" in arg
                        )
                        self.assertIs(parsed["save"], expected)

                    with (
                        patch("molo.__main__.ARGV", arguments),
                        patch("molo.__main__.install"),
                        patch("molo.__main__.entrypoint", side_effect=check_arguments),
                    ):
                        main()
                    self.assertEqual(
                        arguments,
                        ["molo", "save=True", mode, "source=image.jpg", *options],
                    )
