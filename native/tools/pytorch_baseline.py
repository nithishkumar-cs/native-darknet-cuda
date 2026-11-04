"""Matched PyTorch baseline for the native Darknet-cfg detector trainer (speed comparison only).

Builds the same graph as the native framework from a Darknet .cfg (convolutional+BN+leaky,
maxpool, route, upsample, shortcut, reorg, [yolo]/[region] heads), loads and saves Darknet
.weights, and trains with the native recipe: the YOLO/region loss of native/src/yolo_loss.cpp
(best-anchor assignment over all heads, ignore_thresh, box scale 2-w*h, region softmax/rescore,
no per-target normalization), crop/letterbox/flip/HSV augmentation of native/src/train_main.cpp,
SGD with classic momentum and weight decay on conv weights only, Darknet burn-in/step LR, and
batch norm with eps 1e-5 and running-stat momentum 0.01. The CLI mirrors cuda_detector_train
(same positional arguments and flags), writes a Darknet checkpoint plus the same
"<output>.step" sidecar after every epoch, so native/tools/benchmark_training.py times both
identically and native cuda_detector_eval scores the PyTorch checkpoints.

Deviations, all in the PyTorch side's favour or neutral: by default the loader letterboxes in uint8
(torch's 2-tap bilinear) and jitters HSV with PIL lookup tables, so images travel to the GPU as
uint8 within ~1/255 of native's float pixels; --exact-inputs reproduces native's float math
instead (used for the parity check, ~4x more CPU per image). Target assignment runs in the
DataLoader workers so the GPU step has no host syncs; cuDNN autotuning (cudnn.benchmark) is on.
BN running variance uses PyTorch's unbiased batch variance (native: biased), a <0.1% difference.
"""
from __future__ import annotations

import argparse
import functools
import json
import math
import os
import struct
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

HEAD_KINDS = ("yolo", "region")
IMAGE_EXTENSIONS = {".ppm", ".jpg", ".jpeg", ".png"}


def parse_cfg(path):
    """Parses a Darknet .cfg into ordered (section, options) pairs; raises ValueError if malformed."""
    sections = []
    for number, raw in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        line = raw.split("#", 1)[0].split(";", 1)[0].strip()
        if not line:
            continue
        if line.startswith("["):
            if not line.endswith("]"):
                raise ValueError(f"bad cfg section at line {number}")
            sections.append((line[1:-1].strip(), {}))
        elif sections and "=" in line:
            key, value = (part.strip() for part in line.split("=", 1))
            sections[-1][1][key] = value
        else:
            raise ValueError(f"bad cfg option at line {number}")
    if not sections or sections[0][0] != "net":
        raise ValueError("cfg must start with [net]")
    return sections


def _ints(text):
    """Comma-separated integer list."""
    return [int(part) for part in text.split(",") if part.strip()]


class DarknetNet(nn.Module):
    """Darknet-cfg graph as a PyTorch module; forward returns the raw head logits in layer order."""

    def __init__(self, cfg):
        """Builds layers and shape bookkeeping from cfg; raises ValueError on unsupported sections."""
        super().__init__()
        sections = parse_cfg(cfg)
        net = sections[0][1]
        self.channels, self.height, self.width = int(net["channels"]), int(net["height"]), int(net["width"])
        c, h, w = self.channels, self.height, self.width
        self.kinds, self.specs, self.heads = [], [], []
        self.layers = nn.ModuleList()
        shapes = []
        for name, opt in sections[1:]:
            index = len(self.kinds)
            spec, module = {}, nn.Identity()
            if name == "convolutional":
                filters, size, stride = int(opt["filters"]), int(opt["size"]), int(opt.get("stride", 1))
                pad = size // 2 if int(opt.get("pad", 0)) else int(opt.get("padding", 0))
                bn = int(opt.get("batch_normalize", 0)) != 0
                activation = opt.get("activation", "linear")
                if activation not in ("linear", "leaky"):
                    raise ValueError(f"unsupported activation {activation}")
                module = nn.Module()
                module.conv = nn.Conv2d(c, filters, size, stride, pad, bias=not bn)
                module.bn = nn.BatchNorm2d(filters, eps=1e-5, momentum=0.01) if bn else None
                spec = {"leaky": activation == "leaky"}
                c, h, w = filters, (h + 2 * pad - size) // stride + 1, (w + 2 * pad - size) // stride + 1
            elif name == "maxpool":
                size, stride = int(opt["size"]), int(opt["stride"])
                oh, ow = (h - 1) // stride + 1, (w - 1) // stride + 1
                offset = (size - 1) // 2
                # Native/Darknet window: starts at o*stride-offset, out-of-range taps ignored (-inf pad).
                spec = {"size": size, "stride": stride,
                        "pad": (offset, (ow - 1) * stride + size - w - offset,
                                offset, (oh - 1) * stride + size - h - offset)}
                h, w = oh, ow
            elif name == "upsample":
                spec = {"stride": int(opt["stride"])}
                h, w = h * spec["stride"], w * spec["stride"]
            elif name == "route":
                sources = [s + index if s < 0 else s for s in _ints(opt["layers"])]
                spec = {"sources": sources}
                c = sum(shapes[s][0] for s in sources)
                h, w = shapes[sources[0]][1:]
            elif name == "shortcut":
                source = int(opt["from"])
                spec = {"source": source + index if source < 0 else source,
                        "leaky": opt.get("activation", "linear") == "leaky"}
            elif name == "reorg":
                stride = int(opt.get("stride", 2))
                spec = {"stride": stride}
                c, h, w = c * stride * stride, h // stride, w // stride
            elif name in HEAD_KINDS:
                anchors = _ints(opt["anchors"])
                anchors = [(float(anchors[i]), float(anchors[i + 1])) for i in range(0, len(anchors), 2)]
                region = name == "region"
                mask = list(range(int(opt.get("num", 1)))) if region else _ints(opt["mask"])
                classes = int(opt["classes"])
                if c != len(mask) * (classes + 5):
                    raise ValueError(f"head {index}: {c} channels for {len(mask)} anchors x {classes + 5}")
                self.heads.append({
                    "kind": name, "layer": index, "classes": classes, "mask": mask,
                    "anchors": [anchors[m] for m in mask], "h": h, "w": w,
                    "ignore_thresh": float(opt.get("thresh" if region else "ignore_thresh", 0.5 if region else 0.7)),
                    "rescore": region and int(opt.get("rescore", 0)) != 0,
                    "object_scale": float(opt.get("object_scale", 1)) if region else 1.0,
                    "noobject_scale": float(opt.get("noobject_scale", 1)) if region else 1.0,
                    "class_scale": float(opt.get("class_scale", 1)) if region else 1.0,
                    "coord_scale": float(opt.get("coord_scale", 1)) if region else 1.0})
            else:
                raise ValueError(f"unsupported layer [{name}]")
            self.kinds.append(name)
            self.specs.append(spec)
            self.layers.append(module)
            shapes.append((c, h, w))
        self.keep = {s for spec in self.specs for s in spec.get("sources", [])}
        self.keep |= {spec["source"] for spec in self.specs if "source" in spec}
        self.head_layers = {head["layer"] for head in self.heads}

    def initialize(self, seed):
        """Native initialize_random distribution: N(0, sqrt(2/fan_in)) weights, zero biases, unit BN scale."""
        generator = torch.Generator().manual_seed(seed)
        for kind, module in zip(self.kinds, self.layers):
            if kind != "convolutional":
                continue
            weight = module.conv.weight
            fan_in = weight[0].numel()
            with torch.no_grad():
                weight.copy_(torch.randn(weight.shape, generator=generator) * math.sqrt(2.0 / fan_in))
                if module.bn is None:
                    module.conv.bias.zero_()
                else:
                    module.bn.reset_parameters()

    def convs(self):
        """(layer index, module) for every convolutional layer, in Darknet weights-file order."""
        return [(i, m) for i, (k, m) in enumerate(zip(self.kinds, self.layers)) if k == "convolutional"]

    def forward(self, x):
        """Runs the graph; returns a list of raw NCHW head outputs (one per [yolo]/[region])."""
        saved, heads = {}, []
        for index, (kind, spec, module) in enumerate(zip(self.kinds, self.specs, self.layers)):
            if kind == "convolutional":
                x = module.conv(x)
                if module.bn is not None:
                    x = module.bn(x)
                if spec["leaky"]:
                    x = F.leaky_relu(x, 0.1, inplace=True)
            elif kind == "maxpool":
                if any(spec["pad"]):
                    x = F.pad(x, spec["pad"], value=float("-inf"))
                x = F.max_pool2d(x, spec["size"], spec["stride"])
            elif kind == "upsample":
                x = F.interpolate(x, scale_factor=spec["stride"], mode="nearest")
            elif kind == "route":
                parts = [saved[s] for s in spec["sources"]]
                x = parts[0] if len(parts) == 1 else torch.cat(parts, 1)
            elif kind == "shortcut":
                x = x + saved[spec["source"]]
                if spec["leaky"]:
                    x = F.leaky_relu(x, 0.1)
            elif kind == "reorg":
                # Native reorg channel = c*s*s + (y%s)*s + (x%s), which is exactly pixel_unshuffle.
                x = F.pixel_unshuffle(x, spec["stride"])
            if index in self.head_layers:
                heads.append(x)
            if index in self.keep:
                saved[index] = x
        return heads


def _read_darknet_floats(path):
    """Returns the float payload of a Darknet .weights file (header skipped)."""
    with open(path, "rb") as handle:
        major, minor, _ = struct.unpack("<iii", handle.read(12))
        handle.read(8 if major * 10 + minor >= 2 else 4)
        return np.fromfile(handle, dtype="<f4")


def load_darknet(model, path, layer_limit=-1):
    """Loads Darknet weights into model; layer_limit >= 0 loads only layers [0, limit) (Darknet partial).

    Raises ValueError when the file is short or (full load) has trailing data.
    """
    values, offset = _read_darknet_floats(path), 0
    for index, module in model.convs():
        if 0 <= layer_limit <= index:
            break
        params = [module.bn.bias, module.bn.weight, module.bn.running_mean, module.bn.running_var] \
            if module.bn is not None else [module.conv.bias]
        for tensor in params + [module.conv.weight]:
            n = tensor.numel()
            if offset + n > values.size:
                raise ValueError(f"truncated Darknet weights at layer {index}")
            with torch.no_grad():
                tensor.copy_(torch.from_numpy(values[offset:offset + n].copy()).view_as(tensor))
            offset += n
    if layer_limit < 0 and offset != values.size:
        raise ValueError("weights contain trailing data; cfg may not match weights")


def save_darknet(model, path):
    """Writes model in Darknet .weights format (version 0.2.0, seen 0), same layout native saves."""
    chunks = []
    for _, module in model.convs():
        params = [module.bn.bias, module.bn.weight, module.bn.running_mean, module.bn.running_var] \
            if module.bn is not None else [module.conv.bias]
        chunks += [t.detach().float().reshape(-1) for t in params + [module.conv.weight]]
    flat = torch.cat(chunks).cpu().numpy().astype("<f4")
    with open(path, "wb") as handle:
        handle.write(struct.pack("<iiiQ", 0, 2, 0, 0))
        handle.write(flat.tobytes())


def read_image(path):
    """Reads an RGB uint8 HWC image; binary PPM natively, anything else through Pillow."""
    data = bytearray(Path(path).read_bytes())  # writable, so torch.from_numpy can wrap crops
    if data[:2] != b"P6":
        from PIL import Image
        return np.asarray(Image.open(path).convert("RGB"))
    tokens, pos = [], 2
    while len(tokens) < 3:
        while data[pos:pos + 1].isspace():
            pos += 1
        if data[pos:pos + 1] == b"#":
            pos = data.index(b"\n", pos) + 1
            continue
        end = pos
        while not data[end:end + 1].isspace():
            end += 1
        tokens.append(int(data[pos:end]))
        pos = end
    width, height, maximum = tokens
    if maximum != 255:
        raise ValueError(f"{path}: PPM maxval must be 255")
    pos += 2 if data[pos:pos + 2] == b"\r\n" else 1
    return np.frombuffer(data, np.uint8, width * height * 3, pos).reshape(height, width, 3)


def _letterbox_geometry(iw, ih, width, height):
    """Native letterbox placement in float32: (resized_w, resized_h, left, top)."""
    f32 = np.float32
    scale = min(f32(width) / f32(iw), f32(height) / f32(ih))
    rw = max(1, min(width, int(math.floor(float(f32(iw) * scale) + 0.5))))
    rh = max(1, min(height, int(math.floor(float(f32(ih) * scale) + 0.5))))
    return rw, rh, (width - rw) // 2, (height - rh) // 2


def letterbox_u8(image, width, height):
    """Fast uint8 letterbox: native placement, torch's 2-tap half-pixel bilinear, 128-gray canvas.

    Returns (HWC uint8 canvas, resized_w, resized_h, left, top).
    """
    rw, rh, left, top = _letterbox_geometry(image.shape[1], image.shape[0], width, height)
    resized = F.interpolate(torch.from_numpy(image).permute(2, 0, 1)[None], size=(rh, rw), mode="bilinear",
                            align_corners=False, antialias=False)
    canvas = np.full((height, width, 3), 128, np.uint8)
    canvas[top:top + rh, left:left + rw] = resized[0].permute(1, 2, 0).numpy()
    return canvas, rw, rh, left, top


def jitter_hsv_u8(canvas, hue, saturation, exposure):
    """Fast uint8 jitter_hsv: PIL HSV round trip with hue-shift and S/V-scale lookup tables."""
    from PIL import Image
    levels = np.arange(256)
    table = np.concatenate([(levels + int(round(float(hue) * 256))) % 256,
                            np.clip(np.rint(levels * float(saturation)), 0, 255),
                            np.clip(np.rint(levels * float(exposure)), 0, 255)]).astype(np.int64)
    return np.array(Image.fromarray(canvas).convert("HSV").point(table.tolist()).convert("RGB"))


def letterbox(image, width, height):
    """Native detector::letterbox in float32: aspect-preserving half-pixel bilinear resize on a 0.5 canvas.

    Returns (HWC float32 canvas in [0,1], resized_w, resized_h, left, top).
    """
    ih, iw = image.shape[:2]
    f32 = np.float32
    rw, rh, left, top = _letterbox_geometry(iw, ih, width, height)

    def taps(n_out, n_in):
        """Source indices and weights of the half-pixel bilinear map, as native computes them."""
        src = (np.arange(n_out, dtype=f32) + f32(0.5)) * f32(n_in) / f32(n_out) - f32(0.5)
        low = np.floor(src)
        i0 = np.clip(low.astype(np.int64), 0, n_in - 1)
        return i0, np.minimum(i0 + 1, n_in - 1), np.clip(src - low, 0, 1).astype(f32)

    x0, x1, fx = taps(rw, iw)
    y0, y1, fy = taps(rh, ih)
    pixels = image.astype(f32) / f32(255)
    rows = pixels[:, x0] * (1 - fx)[None, :, None] + pixels[:, x1] * fx[None, :, None]
    canvas = np.full((height, width, 3), 0.5, f32)
    canvas[top:top + rh, left:left + rw] = rows[y0] * (1 - fy)[:, None, None] + rows[y1] * fy[:, None, None]
    return canvas, rw, rh, left, top


def jitter_hsv(canvas, hue, saturation, exposure):
    """Native jitter_hsv on an HWC float32 canvas: HSV round trip with hue shift and S/V scaling, clamped."""
    r, g, b = canvas[..., 0], canvas[..., 1], canvas[..., 2]
    top = np.maximum(np.maximum(r, g), b)
    delta = top - np.minimum(np.minimum(r, g), b)
    s = np.where(top > 0, delta / np.where(top > 0, top, 1), 0).astype(np.float32)
    gray = delta < 1e-8
    safe = np.where(gray, 1, delta)
    h = np.where(top == r, np.fmod((g - b) / safe, 6), np.where(top == g, (b - r) / safe + 2, (r - g) / safe + 4)) / 6
    h = np.where(gray, 0, np.where(h < 0, h + 1, h)) + hue
    h -= np.floor(h)
    s = np.clip(s * saturation, 0, 1)
    v = np.clip(top * exposure, 0, 1)
    hh = h * 6
    low = np.floor(hh)
    sector = np.mod(low, 6).astype(np.int64)
    frac = hh - low
    p, q, t = v * (1 - s), v * (1 - s * frac), v * (1 - s * (1 - frac))
    out = np.stack([np.choose(sector, (v, q, p, p, t, v)), np.choose(sector, (t, v, v, q, p, p)),
                    np.choose(sector, (p, p, t, v, v, q))], axis=-1)
    return np.clip(out, 0, 1, out=out).astype(np.float32, copy=False)


class VocDetection(torch.utils.data.Dataset):
    """Native train_main sample pipeline plus CPU-side YOLO target assignment.

    Items are plan tuples from PlanSampler; returns (image, match_index, match_value, truths) where
    match_index rows are (head, slot, cell_y, cell_x, class) and match_value rows are
    (offset_x, offset_y, log_w, log_h, box_scale, truth cx, cy, w, h in input pixels).
    """

    def __init__(self, images, labels_dir, heads, width, height, exact_inputs):
        """Stores the image list, parsed labels and head geometry (picklable for spawn workers)."""
        self.images = [str(p) for p in images]
        self.labels = []
        for image in images:
            path = Path(labels_dir) / (Path(image).stem + ".txt")
            rows = np.loadtxt(path, dtype=np.float32, ndmin=2) if path.stat().st_size else np.zeros((0, 5), np.float32)
            self.labels.append(rows.reshape(-1, 5))
        self.width, self.height, self.exact_inputs = width, height, exact_inputs
        self.anchor_table = [(h, s, aw, ah) for h, head in enumerate(heads) for s, (aw, ah) in enumerate(head["anchors"])]
        self.grids = [(head["h"], head["w"], head["coord_scale"]) for head in heads]

    def __len__(self):
        """Number of images."""
        return len(self.images)

    def __getitem__(self, plan):
        """Loads, crops, letterboxes, flips and color-jitters one image per its plan; assigns targets."""
        index, crop, flip, color = plan
        image = read_image(self.images[index])
        labels = self.labels[index].copy()
        if crop is not None:
            image, labels = self._crop(image, labels, crop)
        resize = letterbox if self.exact_inputs else letterbox_u8
        canvas, rw, rh, left, top = resize(image, self.width, self.height)
        if len(labels):
            labels[:, 1] = (labels[:, 1] * rw + left) / self.width
            labels[:, 2] = (labels[:, 2] * rh + top) / self.height
            labels[:, 3] *= rw / self.width
            labels[:, 4] *= rh / self.height
        if flip:
            canvas = np.ascontiguousarray(canvas[:, ::-1])
            labels[:, 1] = 1 - labels[:, 1]
        if color is not None:
            canvas = (jitter_hsv if self.exact_inputs else jitter_hsv_u8)(canvas, *color)
        if self.exact_inputs:
            canvas = canvas.transpose(2, 0, 1)
        return (torch.from_numpy(np.ascontiguousarray(canvas)), *self._assign(labels))

    @staticmethod
    def _crop(image, labels, fractions):
        """native crop_rect_from + crop_target: crops the image and remaps/clips/drops labels."""
        ih, iw = image.shape[:2]
        left, right = int(fractions[0] * iw), int(fractions[1] * iw)
        top, bottom = int(fractions[2] * ih), int(fractions[3] * ih)
        w, h = max(1, iw - left - right), max(1, ih - top - bottom)
        left, top = min(left, iw - w), min(top, ih - h)
        image = image[top:top + h, left:left + w]
        if not len(labels):
            return image, labels
        cx, cy, bw, bh = (labels[:, i] for i in range(1, 5))
        x1 = np.clip((cx - bw / 2) * iw, left, left + w)
        x2 = np.clip((cx + bw / 2) * iw, left, left + w)
        y1 = np.clip((cy - bh / 2) * ih, top, top + h)
        y2 = np.clip((cy + bh / 2) * ih, top, top + h)
        ncx, ncy = (x1 + x2) / 2 - left, (y1 + y2) / 2 - top
        keep = (x2 > x1) & (y2 > y1) & (ncx >= 0) & (ncx <= w) & (ncy >= 0) & (ncy <= h)
        out = np.stack([labels[:, 0], ncx / w, ncy / h, (x2 - x1) / w, (y2 - y1) / h], 1)[keep]
        return image, out.astype(np.float32)

    def _assign(self, labels):
        """native yolo_loss assignment: best shape-IoU anchor over all heads (first wins ties)."""
        n = len(labels)
        index = np.zeros((n, 5), np.int64)
        value = np.zeros((n, 9), np.float32)
        truths = np.zeros((n, 4), np.float32)
        for row, (cls, cx, cy, tw, th) in enumerate(labels):
            pw, ph = tw * self.width, th * self.height
            best, best_iou = None, -1.0
            for head, slot, aw, ah in self.anchor_table:
                inter = min(aw, pw) * min(ah, ph)
                iou = inter / (aw * ah + pw * ph - inter)
                if iou > best_iou:
                    best, best_iou = (head, slot, aw, ah), iou
            head, slot, aw, ah = best
            gh, gw, coord_scale = self.grids[head]
            gx, gy = cx * gw, cy * gh
            cell_x, cell_y = min(max(int(gx), 0), gw - 1), min(max(int(gy), 0), gh - 1)
            truth = (cx * self.width, cy * self.height, pw, ph)
            index[row] = (head, slot, cell_y, cell_x, int(cls))
            value[row] = (min(max(gx - cell_x, 0.0), 1.0), min(max(gy - cell_y, 0.0), 1.0),
                          math.log(pw / aw), math.log(ph / ah), coord_scale * (2 - tw * th), *truth)
            truths[row] = truth
        return torch.from_numpy(index), torch.from_numpy(value), torch.from_numpy(truths)


class PlanSampler(torch.utils.data.Sampler):
    """Per-epoch shuffle plus augmentation draws (native SamplePlan), made in the main process."""

    def __init__(self, n, seed, crop_jitter, flip_probability, hue, saturation, exposure):
        """Stores the augmentation ranges; one numpy Generator drives every epoch."""
        self.n, self.rng = n, np.random.default_rng(seed)
        self.crop_jitter, self.flip_probability = crop_jitter, flip_probability
        self.hue, self.saturation, self.exposure = hue, saturation, exposure

    def __len__(self):
        """Samples per epoch."""
        return self.n

    def __iter__(self):
        """Yields (index, crop fractions|None, flip, (hue, sat, exposure)|None) for one epoch."""
        rng, n = self.rng, self.n
        order = rng.permutation(n)
        crops = rng.uniform(0, self.crop_jitter, (n, 4)) if self.crop_jitter > 0 else None
        flips = rng.random(n) < self.flip_probability
        color = self.hue > 0 or self.saturation > 0 or self.exposure > 0
        if color:
            hsv = np.stack([rng.uniform(-self.hue, self.hue, n), rng.uniform(1 - self.saturation, 1 + self.saturation, n),
                            rng.uniform(1 - self.exposure, 1 + self.exposure, n)], 1).astype(np.float32)
        for k, index in enumerate(order):
            yield (int(index), None if crops is None else tuple(crops[k]), bool(flips[k]),
                   tuple(hsv[k]) if color else None)


def collate(batch, heads):
    """Stacks images, splits matches per head with the batch index prepended, pads truths to [B,T,4]."""
    images = torch.stack([item[0] for item in batch])
    index = torch.cat([torch.cat([torch.full((len(item[1]), 1), b, dtype=torch.int64), item[1]], 1)
                       for b, item in enumerate(batch)])
    value = torch.cat([item[2] for item in batch])
    per_head = [(index[index[:, 1] == h], value[index[:, 1] == h]) for h in range(heads)]
    t_max = max(1, max(len(item[3]) for item in batch))
    truths = torch.zeros(len(batch), t_max, 4)
    for b, item in enumerate(batch):
        truths[b, :len(item[3])] = item[3]
    return images, per_head, truths, len(index)


class YoloLoss:
    """native/src/yolo_loss.cpp on the GPU for a whole minibatch; returns the summed loss tensor."""

    def __init__(self, heads, width, height, device):
        """Caches per-head anchor tensors and grid offsets on device."""
        self.heads, self.width, self.height = heads, width, height
        self.anchors = [torch.tensor(head["anchors"], device=device) for head in heads]
        self.grid = [(torch.arange(head["w"], device=device, dtype=torch.float32),
                      torch.arange(head["h"], device=device, dtype=torch.float32).view(-1, 1)) for head in heads]
        self.false = torch.zeros((), dtype=torch.bool, device=device)  # device scalar: index_put_ without a host sync

    def __call__(self, outputs, per_head, truths):
        """Loss summed over the batch; per_head[h] = (index [M,6] b/head/slot/cy/cx/cls, value [M,9])."""
        total = 0.0
        tx, ty, tw, th = (truths[..., i] for i in range(4))
        for h, (out, head) in enumerate(zip(outputs, self.heads)):
            batch, attrs = out.shape[0], head["classes"] + 5
            o = out.float().view(batch, len(head["mask"]), attrs, head["h"], head["w"])
            anchors = self.anchors[h]
            gx, gy = self.grid[h]
            with torch.no_grad():
                px = (torch.sigmoid(o[:, :, 0]) + gx) * (self.width / head["w"])
                py = (torch.sigmoid(o[:, :, 1]) + gy) * (self.height / head["h"])
                pw = torch.exp(o[:, :, 2].clamp(-20, 20)) * anchors[:, 0].view(1, -1, 1, 1)
                ph = torch.exp(o[:, :, 3].clamp(-20, 20)) * anchors[:, 1].view(1, -1, 1, 1)
                negative = torch.ones_like(px, dtype=torch.bool)
                if head["ignore_thresh"] < 1.0:
                    iou = _box_iou(px.flatten(1)[..., None], py.flatten(1)[..., None], pw.flatten(1)[..., None],
                                   ph.flatten(1)[..., None], tx[:, None], ty[:, None], tw[:, None], th[:, None])
                    negative &= (iou.amax(2) < head["ignore_thresh"]).view_as(px)
            index, value = per_head[h]
            b, _, slot, cy, cx, cls = index.unbind(1)
            negative.index_put_((b, slot, cy, cx), self.false)
            total = total + head["noobject_scale"] * (F.softplus(o[:, :, 4]) * negative).sum()
            if not len(index):
                continue
            p = o[b, slot, :, cy, cx]
            if head["rescore"]:
                with torch.no_grad():
                    target = _box_iou(px[b, slot, cy, cx], py[b, slot, cy, cx], pw[b, slot, cy, cx],
                                      ph[b, slot, cy, cx], value[:, 5], value[:, 6], value[:, 7], value[:, 8])
            else:
                target = torch.ones_like(p[:, 4])
            total = total + head["object_scale"] * F.binary_cross_entropy_with_logits(p[:, 4], target, reduction="sum")
            scale = value[:, 4:5]
            total = total + (scale * F.binary_cross_entropy_with_logits(p[:, 0:2], value[:, 0:2], reduction="none")).sum()
            total = total + (0.5 * scale * (p[:, 2:4] - value[:, 2:4]).square()).sum()
            if head["kind"] == "region":
                total = total + head["class_scale"] * F.cross_entropy(p[:, 5:], cls, reduction="sum")
            else:
                onehot = torch.zeros_like(p[:, 5:]).scatter_(1, cls[:, None], 1.0)
                total = total + F.binary_cross_entropy_with_logits(p[:, 5:], onehot, reduction="sum")
        return total


def _box_iou(x1, y1, w1, h1, x2, y2, w2, h2):
    """IoU of center-size boxes (broadcasting); zero-size padding boxes give 0 (native box_iou)."""
    ix = (torch.minimum(x1 + w1 / 2, x2 + w2 / 2) - torch.maximum(x1 - w1 / 2, x2 - w2 / 2)).clamp_min(0)
    iy = (torch.minimum(y1 + h1 / 2, y2 + h2 / 2) - torch.maximum(y1 - h1 / 2, y2 - h2 / 2)).clamp_min(0)
    inter = ix * iy
    denom = w1 * h1 + w2 * h2 - inter
    return torch.where(denom > 0, inter / denom.clamp_min(1e-30), torch.zeros_like(inter))


def learning_rate(base, iteration, burn_in, steps, scales):
    """Native LearningRateSchedule::at (burn-in power 4, then step decay)."""
    if burn_in > 0 and iteration < burn_in:
        return base * ((iteration + 1) / burn_in) ** 4
    for step, scale in zip(steps, scales):
        if iteration >= step:
            base *= scale
    return base


def set_precision(precision):
    """fp32 = strict IEEE FP32 like the native build (no TF32); tf32 = PyTorch's cuDNN default; amp = fp16 autocast."""
    strict = precision == "fp32"
    torch.backends.cudnn.allow_tf32 = not strict
    torch.backends.cuda.matmul.allow_tf32 = not strict
    torch.backends.cudnn.benchmark = True


def to_device(batch, device):
    """Non-blocking upload of a collated batch; uint8 NHWC images become float NCHW in [0,1] on the GPU.

    NCHW throughout: channels_last measured ~4x slower for every precision on YOLOv3-tiny here.
    """
    images, per_head, truths, positives = batch
    images = images.to(device, non_blocking=True)
    if images.dtype == torch.uint8:
        images = images.permute(0, 3, 1, 2).float().div_(255).contiguous()
    per_head = [(i.to(device, non_blocking=True), v.to(device, non_blocking=True)) for i, v in per_head]
    return images, per_head, truths.to(device, non_blocking=True), positives


def train(args):
    """cuda_detector_train-compatible training loop; checkpoints Darknet weights + .step every epoch."""
    if Path(args.output).exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.output}")
    if args.backend != "cuda":
        raise SystemExit("the PyTorch baseline trains on CUDA only")
    set_precision(args.precision)
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    model = DarknetNet(args.cfg)
    if args.init:
        if args.init_layers >= 0:
            model.initialize(args.seed)
        load_darknet(model, args.init, args.init_layers)
    else:
        model.initialize(args.seed)
    model.to(device).train()
    images = sorted(p for p in Path(args.images).iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    if args.limit > 0:
        images = images[:args.limit]
    dataset = VocDetection(images, args.labels, model.heads, model.width, model.height, args.exact_inputs)
    sampler = PlanSampler(len(dataset), args.seed, args.crop_jitter, args.flip_probability, args.hue_jitter,
                          args.saturation_jitter, args.exposure_jitter)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch, sampler=sampler,
                                         collate_fn=functools.partial(collate, heads=len(model.heads)),
                                         num_workers=args.workers, pin_memory=True,
                                         persistent_workers=args.workers > 0, prefetch_factor=2 if args.workers else None)
    decayed = [m.conv.weight for _, m in model.convs()]
    ids = {id(p) for p in decayed}
    others = [p for p in model.parameters() if id(p) not in ids]
    optimizer = torch.optim.SGD([{"params": decayed, "weight_decay": args.decay},
                                 {"params": others, "weight_decay": 0.0}], lr=args.lr, momentum=args.momentum)
    criterion = YoloLoss(model.heads, model.width, model.height, device)
    amp = args.precision == "amp"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    print(f"training_backend=pytorch images={len(dataset)} epochs={args.epochs} batch={args.batch} "
          f"precision={args.precision} workers={args.workers} "
          f"torch={torch.__version__} cudnn={torch.backends.cudnn.version()}", flush=True)
    iteration = 0
    for epoch in range(1, args.epochs + 1):
        loss_sum = torch.zeros((), device=device)
        positives = 0
        for batch in loader:
            x, per_head, truths, count = to_device(batch, device)
            lr = learning_rate(args.lr, iteration, args.burn_in, args.steps, args.scales)
            for group in optimizer.param_groups:
                group["lr"] = lr
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                outputs = model(x)
            loss = criterion(outputs, per_head, truths)
            scaler.scale(loss / x.shape[0]).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            loss_sum += loss.detach()
            positives += count
            iteration += 1
        save_darknet(model, args.output)
        Path(args.output + ".step").write_text(f"{iteration} {epoch}\n")
        print(f"epoch={epoch} mean_loss={loss_sum.item() / len(dataset):g} positives={positives} lr={lr:g} "
              f"checkpoint={args.output}", flush=True)


def bench_infer(args):
    """Native `cuda_detector benchmark --scope network` protocol: upload + forward + head download, mean ms."""
    set_precision("fp32" if args.precision == "fp32" else "tf32")
    device = torch.device("cuda")
    model = DarknetNet(args.cfg)
    load_darknet(model, args.weights)
    model.to(device).eval()
    dtype = torch.float16 if args.precision == "fp16" else torch.float32
    model.to(dtype)
    canvas = letterbox(read_image(args.image), model.width, model.height)[0]
    host = torch.from_numpy(np.ascontiguousarray(canvas.transpose(2, 0, 1)))[None].to(dtype)
    times = []
    with torch.inference_mode():
        for i in range(args.warmup + args.iterations):
            start = time.perf_counter()
            outputs = [o.cpu() for o in model(host.to(device))]
            if i >= args.warmup:
                times.append((time.perf_counter() - start) * 1000)
    print(json.dumps({"cfg": args.cfg, "precision": args.precision, "mean_ms": round(float(np.mean(times)), 3),
                      "min_ms": round(float(np.min(times)), 3), "heads": [list(o.shape) for o in outputs],
                      "torch": torch.__version__, "cudnn": torch.backends.cudnn.version()}), flush=True)


def main():
    """CLI: `train` mirrors cuda_detector_train's arguments; `bench-infer` times batch-1 inference."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="command", required=True)
    t = sub.add_parser("train")
    for name in ("cfg", "images", "labels", "output"):
        t.add_argument(name)
    t.add_argument("--init", default="")
    t.add_argument("--init-layers", type=int, default=-1)
    t.add_argument("--epochs", type=int, default=1)
    t.add_argument("--limit", type=int, default=0)
    t.add_argument("--batch", type=int, default=1)
    t.add_argument("--lr", type=float, default=0.0001)
    t.add_argument("--decay", type=float, default=0.0005)
    t.add_argument("--momentum", type=float, default=0.9)
    t.add_argument("--seed", type=int, default=2026)
    t.add_argument("--burn-in", type=int, default=0)
    t.add_argument("--steps", type=_ints, default=[])
    t.add_argument("--scales", type=lambda s: [float(v) for v in s.split(",") if v], default=[])
    for name in ("flip-probability", "exposure-jitter", "saturation-jitter", "hue-jitter", "crop-jitter"):
        t.add_argument(f"--{name}", type=float, default=0.0)
    t.add_argument("--backend", default="cuda")
    t.add_argument("--precision", choices=["fp32", "tf32", "amp"], default="fp32")
    # Each spawned worker imports torch (~300 MB); 16 exhausted this 16 GB laptop's RAM and then
    # pinned-memory allocation failed, while 8 already sustain ~630 img/s of augmented YOLOv3-tiny input.
    t.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 1)))
    t.add_argument("--exact-inputs", action="store_true", help="ship float32 inputs (bit-faithful to native)")
    b = sub.add_parser("bench-infer")
    for name in ("cfg", "weights", "image"):
        b.add_argument(name)
    b.add_argument("--precision", choices=["fp32", "tf32", "fp16"], default="fp32")
    b.add_argument("--warmup", type=int, default=10)
    b.add_argument("--iterations", type=int, default=100)
    args = p.parse_args()
    if args.command == "train":
        if len(args.steps) != len(args.scales):
            raise SystemExit("--steps and --scales must have the same length")
        train(args)
    else:
        bench_infer(args)


if __name__ == "__main__":
    main()
