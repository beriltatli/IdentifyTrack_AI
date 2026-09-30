"""Baseline 1: greedy IoU. No motion model, no appearance, memory of exactly one frame.

Every detection is emitted with an ID (the contract shared by all trackers, which is what
keeps DetA identical across them): it inherits the ID of the best-overlapping box from the
previous frame, or gets a fresh one.
"""
from __future__ import annotations

import numpy as np

from detection.types import Detections
from eval.common import Frame, Sequence, empty_frame, iou_matrix


def run_greedy_iou(dets: list[Detections], iou_thr: float = 0.3) -> Sequence:
    out: Sequence = []
    prev_ids = np.empty(0, dtype=np.int64)
    prev_boxes = np.empty((0, 4))
    next_id = 1
    for d in dets:
        n = len(d)
        ids = np.zeros(n, dtype=np.int64)
        if n and len(prev_ids):
            iou = iou_matrix(prev_boxes, d.boxes)
            taken_p, taken_d = set(), set()
            for flat in np.argsort(-iou, axis=None):  # best overlap first
                p, j = divmod(int(flat), n)
                if iou[p, j] < iou_thr:
                    break
                if p in taken_p or j in taken_d:
                    continue
                ids[j] = prev_ids[p]
                taken_p.add(p)
                taken_d.add(j)
        for j in range(n):
            if ids[j] == 0:
                ids[j] = next_id
                next_id += 1
        out.append(Frame(ids, d.boxes.copy()) if n else empty_frame())
        prev_ids, prev_boxes = ids, d.boxes
    return out
