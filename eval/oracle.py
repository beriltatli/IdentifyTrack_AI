"""Oracle detections: feed ground truth to the tracker instead of a detector.

This removes detection error entirely, so the AssA that comes out is pure association
quality. One subtlety decides whether the oracle is meaningful: MOT17 keeps annotating a
person who is fully behind a pillar. A detector cannot see through the pillar, so the oracle
withholds boxes of objects whose visibility is below `min_visibility` (the same `hidden` set
used for occlusion gaps). Without this the tracker would never lose anything, there would be
no gaps to recover, and the oracle would report a flattering nothing.
"""
from __future__ import annotations

import numpy as np

from detection.types import Detections
from eval.common import Sequence


def oracle_detections(
    gt: Sequence, hidden: set[tuple[int, int]] | None = None
) -> list[Detections]:
    out: list[Detections] = []
    for t, f in enumerate(gt):
        keep = np.array([not (hidden and (int(i), t) in hidden) for i in f.ids], dtype=bool)
        boxes = f.boxes[keep] if len(f.ids) else np.empty((0, 4))
        out.append(Detections(boxes, np.ones(len(boxes))))
    return out
