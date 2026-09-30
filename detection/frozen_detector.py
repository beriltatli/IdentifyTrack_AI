"""The detector, as a black box. Its weights and thresholds are pinned in config.yaml and are
never trained or tuned here. Two interchangeable backends produce the same thing: per-frame
boxes + scores above `conf_floor` (everything else is the tracker's business).

  mot17_public  the benchmark authors' det.txt (SDP). Zero extra dependencies.
  yolo          ultralytics YOLO on the raw frames (needs torch).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from detection.types import Detections, empty_detections
from eval.mot_io import seq_length


def load_mot17_public(seq_dir: Path, conf_floor: float) -> list[Detections]:
    n = seq_length(seq_dir)
    rows = np.loadtxt(seq_dir / "det" / "det.txt", delimiter=",", ndmin=2)
    rows = rows[rows[:, 6] >= conf_floor]
    out: list[Detections] = []
    for f in range(1, n + 1):
        r = rows[rows[:, 0] == f]
        if not len(r):
            out.append(empty_detections())
            continue
        boxes = np.stack([r[:, 2], r[:, 3], r[:, 2] + r[:, 4], r[:, 3] + r[:, 5]], axis=1)
        out.append(Detections(boxes.astype(np.float64), r[:, 6].astype(np.float64)))
    return out


class YoloDetector:
    """Frozen ultralytics YOLO. Imported lazily so the rest of the repo works without torch."""

    def __init__(self, weights: str, imgsz: int, conf_floor: float, person_class: int = 0):
        from ultralytics import YOLO  # noqa: PLC0415

        import torch  # noqa: PLC0415

        self.model = YOLO(weights)
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.imgsz, self.conf, self.cls = imgsz, conf_floor, person_class

    def __call__(self, image_paths: list[Path]) -> list[Detections]:
        out = []
        for p in image_paths:
            r = self.model.predict(str(p), imgsz=self.imgsz, conf=self.conf, classes=[self.cls],
                                   device=self.device, verbose=False)[0]
            b = r.boxes
            out.append(Detections(b.xyxy.cpu().numpy().astype(np.float64),
                                  b.conf.cpu().numpy().astype(np.float64)))
        return out


def save_dets(dets: list[Detections], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    counts = np.array([len(d) for d in dets])
    n = int(counts.sum())
    boxes = np.concatenate([d.boxes for d in dets]) if n else np.empty((0, 4))
    scores = np.concatenate([d.scores for d in dets]) if n else np.empty(0)
    extra = {}
    if dets and dets[0].feats is not None and n:
        extra["feats"] = np.concatenate([d.feats for d in dets if len(d)])
    np.savez_compressed(path, counts=counts, boxes=boxes, scores=scores, **extra)


def load_dets(path: Path) -> list[Detections]:
    z = np.load(path)
    edges = np.concatenate([[0], np.cumsum(z["counts"])])
    feats = z["feats"] if "feats" in z.files else None
    return [
        Detections(z["boxes"][a:b], z["scores"][a:b], None if feats is None else feats[a:b])
        for a, b in zip(edges[:-1], edges[1:])
    ]
