"""Baseline 3: appearance only. Cosine distance on ReID features, no motion model at all.

Each track keeps a gallery of past embeddings (same Gallery class as the full tracker, always
updated -- there is no occlusion reasoning here) and stays matchable for `max_age` frames.
Position is never consulted, so two similar-looking people anywhere in the frame can swap.
Emission contract as everywhere: every detection >= output_conf is emitted with an id.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from detection.types import Detections
from eval.common import Frame, Sequence, empty_frame
from tracker.assignment import linear_sum_assignment
from tracker.gallery import Gallery


@dataclass
class _T:
    id: int
    gallery: Gallery
    misses: int = 0


def run_appearance_only(
    dets: list[Detections], output_conf: float = 0.30, app_gate: float = 0.5,
    max_age: int = 100, gallery_size: int = 30,
) -> Sequence:
    tracks: list[_T] = []
    next_id = 1
    out: Sequence = []
    for d in dets:
        if d.feats is None:
            raise ValueError("appearance-only baseline needs ReID features")
        high = np.flatnonzero(d.scores >= output_conf)
        ids = np.zeros(len(d), dtype=np.int64)
        seen: set[int] = set()
        if tracks and len(high):
            dist = np.stack([t.gallery.distance(d.feats[high]) for t in tracks])
            cost = np.where(dist <= app_gate, dist, np.inf)
            for ti, hj in zip(*linear_sum_assignment(cost)):
                j = int(high[hj])
                tracks[ti].gallery.update(d.feats[j], d.scores[j], 0.0)
                ids[j] = tracks[ti].id
                seen.add(ti)
        for ti, t in enumerate(tracks):
            t.misses = 0 if ti in seen else t.misses + 1
        for j in high:
            if ids[j] == 0:
                g = Gallery(gallery_size, "always")
                g.seed(d.feats[j])
                tracks.append(_T(next_id, g))
                ids[j] = next_id
                next_id += 1
        tracks = [t for t in tracks if t.misses <= max_age]
        out.append(Frame(ids[high].copy(), d.boxes[high].copy()) if len(high) else empty_frame())
    return out
