"""CLEAR-MOT matching, MOTA (legacy column) and the raw identity-switch events.

MOTA = 1 - (FN + FP + IDSW) / num_gt. An ID switch costs exactly one missed box, and FN/FP
come from the detector -- which is why MOTA is not this repo's headline. The per-frame
matches computed here are also the shared substrate for gaps.py and taxonomy.py, so all
"switch" counts in the repo agree with each other by construction.

Matching: IoU >= 0.5, and a GT track prefers the pred ID it was last matched to (continuity
bonus) before falling back to a global optimum. The last-matched ID persists across frames
where the GT is missing/unmatched, so returning with a different ID is counted as a switch.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from eval.common import IOU_MATCH, Sequence, iou_matrix
from tracker.assignment import linear_sum_assignment

_CONTINUITY_BONUS = 1000.0


@dataclass(frozen=True)
class SwitchEvent:
    frame: int  # frame where the GT track is matched to its new pred ID
    gt_id: int
    old_pred: int
    new_pred: int
    prev_frame: int  # frame of the last match to old_pred


@dataclass(frozen=True)
class ClearResult:
    mota: float
    fp: int
    fn: int
    idsw: int
    num_gt: int
    matches: list[list[tuple[int, int]]]  # per frame: [(gt_id, pred_id), ...]
    switches: list[SwitchEvent]


def clear_match(gt: Sequence, pred: Sequence, iou_thr: float = IOU_MATCH) -> ClearResult:
    if len(gt) != len(pred):
        raise ValueError("gt and pred must cover the same number of frames")
    last_pred: dict[int, int] = {}
    last_frame: dict[int, int] = {}
    fp = fn = idsw = num_gt = 0
    matches: list[list[tuple[int, int]]] = []
    switches: list[SwitchEvent] = []
    for t, (g, p) in enumerate(zip(gt, pred)):
        num_gt += len(g.ids)
        pairs: list[tuple[int, int]] = []
        if len(g.ids) and len(p.ids):
            sim = iou_matrix(g.boxes, p.boxes)
            ok = sim >= iou_thr - 1e-9
            prev = np.array([last_pred.get(int(i), -1) for i in g.ids])
            bonus = (prev[:, None] == p.ids[None, :]) * _CONTINUITY_BONUS
            cost = np.where(ok, -(sim + bonus), np.inf)
            rows, cols = linear_sum_assignment(cost)
            for r, c in zip(rows, cols):
                gid, pid = int(g.ids[r]), int(p.ids[c])
                pairs.append((gid, pid))
                if gid in last_pred and last_pred[gid] != pid:
                    idsw += 1
                    switches.append(SwitchEvent(t, gid, last_pred[gid], pid, last_frame[gid]))
                last_pred[gid], last_frame[gid] = pid, t
        tp = len(pairs)
        fn += len(g.ids) - tp
        fp += len(p.ids) - tp
        matches.append(pairs)
    mota = 1.0 - (fn + fp + idsw) / num_gt if num_gt else 0.0
    return ClearResult(mota, fp, fn, idsw, num_gt, matches, switches)
