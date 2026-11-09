# Copyright (C) 2026 RidgeRun, LLC (http://www.ridgerun.com)
# SPDX-License-Identifier: AGPL-3.0-only

"""Nearest-neighbor upsampling for MAX Graph."""

from max import nn
from max.graph import TensorValue, ops


class UpsampleNearest2x(nn.Module):
    """Double the height and width of an NCHW tensor."""

    def __call__(self, x: TensorValue) -> TensorValue:
        x = ops.repeat_interleave(x, repeats=2, axis=2, out_dim=x.shape[2] * 2)
        return ops.repeat_interleave(x, repeats=2, axis=3, out_dim=x.shape[3] * 2)
