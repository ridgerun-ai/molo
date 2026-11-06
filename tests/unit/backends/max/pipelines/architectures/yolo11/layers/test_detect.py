import unittest
import numpy as np
import torch
from ultralytics.nn.modules.head import Detect as Reference
from max.driver import Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import Graph, TensorType
from molo.backends.max.pipelines.architectures.yolo11.layers.detect import YOLO11Detect
from tests.unit.helpers import DEVICE, REF, prepare


class DetectTest(unittest.TestCase):
    def test_raw_and_decoded(self):
        self.check_head(YOLO11Detect, Reference)

    def check_head(self, model_type, reference_type):
        for classes, channels, shapes in [
            (5, (8, 16, 32), ((1, 8, 4, 6), (1, 16, 2, 3), (1, 32, 1, 2))),
            (2, (8, 16, 32), ((1, 8, 2, 2), (1, 16, 1, 1), (1, 32, 1, 1))),
        ]:
            with self.subTest(classes=classes, shapes=shapes):
                reference = prepare(reference_type(nc=classes, ch=channels))
                reference.stride = torch.tensor([8.0, 16.0, 32.0])
                weights = {
                    f"layer.{k}": v.detach().numpy()
                    for k, v in reference.state_dict().items()
                    if not k.endswith("num_batches_tracked") and k != "dfl.conv.weight"
                }
                model = model_type(classes, channels, device=REF, name="layer")
                with Graph(
                    "detect_test",
                    input_types=[TensorType(DType.float32, s, REF) for s in shapes],
                ) as graph:
                    raw = model.raw([x.tensor for x in graph.inputs])
                    graph.output(model.decode(raw), *raw)
                compiled = InferenceSession(devices=[DEVICE]).load(
                    graph, weights_registry=weights
                )
                inputs = [
                    np.random.default_rng(i).normal(size=s).astype(np.float32)
                    for i, s in enumerate(shapes)
                ]
                actual = [
                    v.to_numpy()
                    for v in compiled.execute(
                        *[Buffer.from_numpy(x).to(DEVICE) for x in inputs]
                    )
                ]
                with torch.inference_mode():
                    features = [torch.from_numpy(x) for x in inputs]
                    raw_ref = [
                        torch.cat((reference.cv2[i](x), reference.cv3[i](x)), dim=1)
                        for i, x in enumerate(features)
                    ]
                    decoded, _ = reference(features)
                for a, b in zip(
                    actual, [decoded.numpy(), *[x.numpy() for x in raw_ref]]
                ):
                    np.testing.assert_allclose(a, b, atol=0.0002, rtol=4e-5)
