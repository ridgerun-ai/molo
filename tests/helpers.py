# Copyright (C) 2026 RidgeRun, LLC (http://www.ridgerun.com)
# SPDX-License-Identifier: AGPL-3.0-only

"""Helpers for comparing MAX modules with Ultralytics on CPU."""

from molo.backends.max.pipelines.architectures.yolo.convert import assign_module
from ultralytics.nn.modules.conv import Conv
from ultralytics.utils.torch_utils import fuse_conv_and_bn

import numpy as np
import torch
from max.driver import CPU, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType

DEVICE = CPU()
REF = DeviceRef.CPU()


def prepare(reference, seed=123, eps=0.001):
    """Prepare a fused inference reference with nontrivial BatchNorm statistics."""
    rng = np.random.default_rng(seed)
    reference.eval()
    for module in reference.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eps = eps
            c = module.num_features
            if module.affine:
                module.weight.data.copy_(
                    torch.from_numpy(rng.uniform(0.5, 1.5, c).astype(np.float32))
                )
                module.bias.data.copy_(
                    torch.from_numpy(rng.normal(0, 0.1, c).astype(np.float32))
                )
            module.running_mean.copy_(
                torch.from_numpy(rng.normal(0, 0.1, c).astype(np.float32))
            )
            module.running_var.copy_(
                torch.from_numpy(rng.uniform(0.5, 1.5, c).astype(np.float32))
            )
    for module in reference.modules():
        if isinstance(module, Conv):
            module.conv = fuse_conv_and_bn(module.conv, module.bn)
            del module.bn
            module.forward = module.forward_fuse
    return reference


def compare(model, reference, shape, seed=17, prefix="", atol=1e-5, rtol=3e-5):
    """Compile and compare a single MAX module with its PyTorch counterpart."""
    reference = prepare(reference)
    weights = {}
    assign_module(weights, prefix, reference)
    model.load_state_dict(weights, strict=True)
    registered_names = {weight.name for weight in model.raw_state_dict().values()}
    assert registered_names == weights.keys(), registered_names ^ weights.keys()
    with Graph(
        "layer_test", input_types=[TensorType(DType.float32, shape, REF)]
    ) as graph:
        graph.output(model(graph.inputs[0].tensor))
    compiled = InferenceSession(devices=[DEVICE]).load(graph, weights_registry=weights)
    x = np.random.default_rng(seed).normal(size=shape).astype(np.float32)
    actual = compiled.execute(Buffer.from_numpy(x).to(DEVICE))[0].to_numpy()
    with torch.inference_mode():
        expected = reference(torch.from_numpy(x)).detach().numpy()
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol)


def module_paths(model, prefix=""):
    """Return every registered MAX module path, including shared children."""
    paths = set()
    for name, child in model.sublayers.items():
        path = f"{prefix}.{name}" if prefix else name
        paths.add(path)
        paths.update(module_paths(child, path))
    return paths
