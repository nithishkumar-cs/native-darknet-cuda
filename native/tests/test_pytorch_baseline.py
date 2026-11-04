"""Checks the PyTorch baseline against this project's reference semantics (CPU torch is enough).

Forward pass: identical head outputs to ONNX Runtime on export_darknet_onnx's graph for the same
Darknet weights (conv/BN/leaky, both maxpool strides, strided conv, shortcut, route, upsample,
reorg, [yolo] and [region]). Weights: Darknet byte-identical round trip and partial/strict loads.
Loss: gradients equal native yolo_loss.cpp's closed forms at zero logits.
"""

import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import pytorch_baseline as pb  # noqa: E402
from export_darknet_onnx import export  # noqa: E402

YOLO_CFG = """[net]
channels=3
height=16
width=16
[convolutional]
batch_normalize=1
filters=8
size=3
stride=1
pad=1
activation=leaky
[maxpool]
size=2
stride=2
[convolutional]
batch_normalize=1
filters=8
size=3
stride=2
pad=1
activation=leaky
[convolutional]
batch_normalize=1
filters=8
size=1
stride=1
pad=1
activation=leaky
[shortcut]
from=-2
activation=linear
[maxpool]
size=2
stride=1
[convolutional]
filters=12
size=1
stride=1
pad=1
activation=linear
[yolo]
mask=2,3
anchors=2,2, 4,4, 6,6, 9,9
classes=1
[route]
layers=-4
[upsample]
stride=2
[route]
layers=-1,1
[convolutional]
filters=12
size=1
stride=1
pad=1
activation=linear
[yolo]
mask=0,1
anchors=2,2, 4,4, 6,6, 9,9
classes=1
"""

REGION_CFG = """[net]
channels=3
height=16
width=16
[convolutional]
batch_normalize=1
filters=8
size=3
stride=1
pad=1
activation=leaky
[maxpool]
size=2
stride=2
[convolutional]
batch_normalize=1
filters=8
size=3
stride=1
pad=1
activation=leaky
[maxpool]
size=2
stride=2
[route]
layers=-2
[reorg]
stride=2
[route]
layers=-1,-3
[convolutional]
filters=14
size=1
stride=1
pad=1
activation=linear
[region]
anchors=3,3, 8,8
num=2
classes=2
rescore=1
object_scale=5
noobject_scale=1
class_scale=1
coord_scale=1
thresh=0.6
"""


def random_model(cfg, seed):
    """DarknetNet with random weights and non-trivial BN statistics."""
    model = pb.DarknetNet(cfg)
    model.initialize(seed)
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for _, module in model.convs():
            if module.bn is not None:
                for tensor, low in ((module.bn.weight, 0.5), (module.bn.bias, -0.5), (module.bn.running_mean, -0.5),
                                    (module.bn.running_var, 0.5)):
                    tensor.copy_(low + torch.rand(tensor.shape, generator=generator))
            else:
                module.conv.bias.copy_(torch.rand(module.conv.bias.shape, generator=generator) - 0.5)
    return model.eval()


class PytorchBaselineTest(unittest.TestCase):
    """Forward/weights/loss checks of native/tools/pytorch_baseline.py."""

    def setUp(self):
        """Temporary directory with both cfgs written out."""
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.cfgs = {}
        for name, text in (("yolo", YOLO_CFG), ("region", REGION_CFG)):
            self.cfgs[name] = self.root / f"{name}.cfg"
            self.cfgs[name].write_text(text, encoding="utf-8")

    def tearDown(self):
        """Removes the temporary directory."""
        self.directory.cleanup()

    def test_forward_matches_onnx_export(self):
        """Every head equals ONNX Runtime's output for the same weights and input."""
        for name, cfg in self.cfgs.items():
            model = random_model(cfg, 3)
            weights, onnx_path = self.root / f"{name}.weights", self.root / f"{name}.onnx"
            pb.save_darknet(model, weights)
            export(cfg, weights, onnx_path)
            x = np.random.default_rng(5).random((1, 3, 16, 16), dtype=np.float32)
            session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
            expected = session.run(None, {session.get_inputs()[0].name: x})
            with torch.no_grad():
                got = [head.numpy() for head in model(torch.from_numpy(x))]
            self.assertEqual(len(got), len(expected), name)
            for e, g in zip(expected, got):
                np.testing.assert_allclose(g, e, rtol=1e-4, atol=1e-5, err_msg=name)

    def test_weights_round_trip_and_partial_load(self):
        """Save/load is byte-identical; partial loads stop early; short or long files are rejected."""
        cfg = self.cfgs["yolo"]
        first, second = self.root / "a.weights", self.root / "b.weights"
        pb.save_darknet(random_model(cfg, 1), first)
        loaded = pb.DarknetNet(cfg)
        pb.load_darknet(loaded, first)
        pb.save_darknet(loaded, second)
        self.assertEqual(first.read_bytes(), second.read_bytes())
        partial = random_model(cfg, 2)
        pb.load_darknet(partial, first, layer_limit=3)
        self.assertTrue(torch.equal(partial.layers[2].conv.weight, loaded.layers[2].conv.weight))
        self.assertFalse(torch.equal(partial.layers[3].conv.weight, loaded.layers[3].conv.weight))
        payload = first.read_bytes()
        for broken in (payload[:-4], payload + b"\0\0\0\0"):
            second.write_bytes(broken)
            with self.assertRaises(ValueError):
                pb.load_darknet(pb.DarknetNet(cfg), second)

    def loss_gradients(self, name, labels):
        """Gradients of the baseline loss w.r.t. all-zero head logits for one image's labels."""
        model = pb.DarknetNet(self.cfgs[name])
        dataset = pb.VocDetection([], self.root, model.heads, 16, 16, True)
        index, value, truths = dataset._assign(np.array(labels, np.float32))
        batch = pb.collate([(torch.zeros(3, 16, 16), index, value, truths)], heads=len(model.heads))
        outputs = [torch.zeros(1, len(h["mask"]) * (h["classes"] + 5), h["h"], h["w"], requires_grad=True)
                   for h in model.heads]
        loss = pb.YoloLoss(model.heads, 16, 16, torch.device("cpu"))(outputs, batch[1], batch[2])
        loss.backward()
        return [o.grad.view(len(h["mask"]), h["classes"] + 5, h["h"], h["w"]) for o, h in zip(outputs, model.heads)]

    def test_yolo_loss_gradients(self):
        """Matched x/y: 0.5*(2-w*h) at a zero offset; w/h: -(2-w*h)*log(t/anchor); obj/class: -0.5; free negatives 0.5."""
        grads = self.loss_gradients("yolo", [[0, 0.0, 0.0, 0.5, 0.5]])
        # An 8x8-pixel box best matches anchor 9x9: head 0 ([yolo] mask=2,3, 4x4 grid), slot 1, cell (0,0).
        g = grads[0][1, :, 0, 0]
        scale = 2 - 0.25
        self.assertAlmostEqual(g[0].item(), 0.5 * scale, places=6)
        self.assertAlmostEqual(g[1].item(), 0.5 * scale, places=6)
        self.assertAlmostEqual(g[2].item(), -scale * math.log(8 / 9), places=5)
        self.assertAlmostEqual(g[4].item(), -0.5, places=6)
        self.assertAlmostEqual(g[5].item(), -0.5, places=6)
        self.assertAlmostEqual(grads[0][0, 4, 3, 3].item(), 0.5, places=6)
        self.assertEqual(grads[1][:, :4].abs().sum().item(), 0.0)
        self.assertTrue(torch.all((grads[1][:, 4] == 0.5) | (grads[1][:, 4] == 0.0)))

    def test_region_loss_gradients(self):
        """Region: softmax class gradient p-onehot, rescored objectness 5*(0.5-IoU), coord scale."""
        grads = self.loss_gradients("region", [[1, 0.5, 0.5, 0.5, 0.5]])
        # 8x8-pixel box: anchor 8,8 (slot 1) at grid cell (2,2) of the 4x4 grid, offset 0. The zero-logit
        # prediction there is an 8x8 box centred at (10,10) against the truth at (8,8): IoU 36/92.
        g = grads[0][1, :, 2, 2]
        iou = 36 / (64 + 64 - 36)
        self.assertAlmostEqual(g[4].item(), 5 * (0.5 - iou), places=5)
        self.assertAlmostEqual(g[5].item(), 0.5, places=6)
        self.assertAlmostEqual(g[6].item(), -0.5, places=6)
        self.assertAlmostEqual(g[0].item(), 0.5 * 1.75, places=6)


if __name__ == "__main__":
    unittest.main()
