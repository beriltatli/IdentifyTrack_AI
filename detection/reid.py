"""Appearance embeddings for detection crops. A commodity, like the detector: not trained here.

  color_hist  hand-made, no torch: per horizontal stripe of the box, a chromaticity histogram
              (illumination-robust hue) plus a brightness histogram, square-rooted (Hellinger)
              and L2-normalised so cosine distance = Bhattacharyya-style similarity. Weak but
              honest; it cannot tell apart two people in similar clothes.
  resnet18    ImageNet-pretrained torchvision backbone, global-pooled 512-d. Needs torch.
              ImageNet features are not person-ReID features; expect modest discrimination.

All embeddings are unit-norm so `feats @ feats.T` is cosine similarity.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Iterator

import numpy as np

from detection.types import Detections


def iter_frames(seq_dir: Path, scale_w: int = 960) -> Iterator[tuple[np.ndarray, float]]:
    """Yield (rgb frame resized to width scale_w, scale factor) via ffmpeg (no OpenCV needed)."""
    kv = dict(l.split("=") for l in (seq_dir / "seqinfo.ini").read_text().split() if "=" in l)
    w0, h0 = int(kv["imWidth"]), int(kv["imHeight"])
    w, h = scale_w, int(round(scale_w * h0 / w0))
    proc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", str(seq_dir / "img1/%06d.jpg"), "-vf", f"scale={w}:{h}",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    size = w * h * 3
    while True:
        buf = proc.stdout.read(size)
        if len(buf) < size:
            break
        yield np.frombuffer(buf, np.uint8).reshape(h, w, 3), w / w0
    proc.stdout.close()
    proc.wait()


class ColorHistExtractor:
    def __init__(self, stripes: int = 3, bins: int = 8) -> None:
        self.stripes, self.bins = stripes, bins
        self.dim = stripes * (bins * bins + bins)

    def __call__(self, frame: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        H, W = frame.shape[:2]
        b = self.bins
        out = np.zeros((len(boxes), self.dim))
        for k, (x1, y1, x2, y2) in enumerate(boxes):
            w = x2 - x1
            x1, x2 = x1 + 0.1 * w, x2 - 0.1 * w  # drop edges: mostly background
            xa, xb = int(np.clip(x1, 0, W - 1)), int(np.clip(x2, 1, W))
            ya, yb = int(np.clip(y1, 0, H - 1)), int(np.clip(y2, 1, H))
            if xb - xa < 2 or yb - ya < self.stripes:
                continue
            crop = frame[ya:yb, xa:xb].astype(np.float32)
            parts = []
            for s in np.array_split(crop, self.stripes, axis=0):
                px = s.reshape(-1, 3)
                tot = px.sum(1) + 1e-6
                r = np.clip((px[:, 0] / tot * b).astype(int), 0, b - 1)
                g = np.clip((px[:, 1] / tot * b).astype(int), 0, b - 1)
                v = np.clip((tot / 3 / 256 * b).astype(int), 0, b - 1)
                h2 = np.bincount(r * b + g, minlength=b * b).astype(np.float64)
                h1 = np.bincount(v, minlength=b).astype(np.float64)
                parts += [h2 / max(h2.sum(), 1), h1 / max(h1.sum(), 1)]
            f = np.sqrt(np.concatenate(parts))
            out[k] = f / max(np.linalg.norm(f), 1e-12)
        return out


class Resnet18Extractor:
    def __init__(self, input_hw: tuple[int, int] = (256, 128)) -> None:
        import torch  # noqa: PLC0415
        import torchvision  # noqa: PLC0415

        self.torch = torch
        m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
        m.fc = torch.nn.Identity()
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.model = m.eval().to(self.device)
        self.hw = input_hw
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
        self.dim = 512

    def __call__(self, frame: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        torch = self.torch
        H, W = frame.shape[:2]
        crops = []
        for x1, y1, x2, y2 in boxes:
            xa, xb = int(np.clip(x1, 0, W - 1)), int(np.clip(x2, 1, W))
            ya, yb = int(np.clip(y1, 0, H - 1)), int(np.clip(y2, 1, H))
            c = torch.from_numpy(frame[ya : max(yb, ya + 2), xa : max(xb, xa + 2)].copy())
            c = c.permute(2, 0, 1).float().unsqueeze(0) / 255
            crops.append(torch.nn.functional.interpolate(c, size=self.hw, mode="bilinear"))
        if not crops:
            return np.zeros((0, self.dim))
        with torch.no_grad():
            batch = (torch.cat(crops).to(self.device) - self.mean) / self.std
            f = torch.nn.functional.normalize(self.model(batch), dim=1)
        return f.cpu().numpy().astype(np.float64)


def make_extractor(cfg: dict):
    if cfg["backend"] == "color_hist":
        return ColorHistExtractor(cfg.get("stripes", 3), cfg.get("bins", 8))
    if cfg["backend"] == "resnet18":
        return Resnet18Extractor()
    raise ValueError(f"unknown reid backend {cfg['backend']!r}")


def attach_features(seq_dir: Path, dets: list[Detections], extractor) -> list[Detections]:
    out = []
    for (frame, sc), d in zip(iter_frames(seq_dir), dets):
        feats = extractor(frame, d.boxes * sc) if len(d) else np.zeros((0, extractor.dim))
        out.append(Detections(d.boxes, d.scores, feats))
    return out
