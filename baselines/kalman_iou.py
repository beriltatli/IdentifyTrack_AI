"""Baseline 2: Kalman + IoU (a SORT-equivalent, written from the description).

Motion only. A track predicts its box with a constant-velocity Kalman filter, detections
are matched to predicted boxes by IoU with the Hungarian solver, and an unmatched confirmed
track keeps coasting for up to `max_age` frames -- the same lost-buffer the full tracker gets,
so the comparison on long gaps isolates what appearance memory adds.

Emission contract (shared by every tracker): each detection is output with an ID, the raw
detector box, never the filtered one, so DetA cannot move. Unconfirmed tracks are therefore
emitted too; they simply die on their first miss.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from detection.types import Detections
from eval.common import Frame, Sequence, empty_frame, iou_matrix
from tracker.assignment import linear_sum_assignment
from tracker.kalman import KalmanBoxFilter


@dataclass
class _Track:
    id: int
    kf: KalmanBoxFilter
    hits: int = 1
    misses: int = 0  # consecutive frames without a match


def run_kalman_iou(
    dets: list[Detections], iou_thr: float = 0.3, max_age: int = 100, min_hits: int = 3
) -> Sequence:
    tracks: list[_Track] = []
    next_id = 1
    out: Sequence = []
    for d in dets:
        n = len(d)
        preds = np.array([t.kf.predict() for t in tracks]) if tracks else np.empty((0, 4))
        ids = np.zeros(n, dtype=np.int64)
        matched_t: set[int] = set()
        if n and len(tracks):
            iou = iou_matrix(preds, d.boxes)
            cost = np.where(iou >= iou_thr, -iou, np.inf)  # gate first, then optimise
            for ti, dj in zip(*linear_sum_assignment(cost)):
                tr = tracks[ti]
                tr.kf.update(d.boxes[dj])
                tr.hits += 1
                tr.misses = 0
                ids[dj] = tr.id
                matched_t.add(int(ti))
        for ti, tr in enumerate(tracks):
            if ti not in matched_t:
                tr.misses += 1
        for j in range(n):
            if ids[j] == 0:
                tracks.append(_Track(next_id, KalmanBoxFilter(d.boxes[j])))
                ids[j] = next_id
                next_id += 1
        tracks = [
            t for t in tracks
            if t.misses == 0 or (t.hits >= min_hits and t.misses <= max_age)
        ]
        out.append(Frame(ids, d.boxes.copy()) if n else empty_frame())
    return out
