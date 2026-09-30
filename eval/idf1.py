"""IDF1 (Ristani et al. 2016): identity-level F1 under a single global GT<->pred track matching.

Unlike CLEAR/MOTA, which counts a switch once at the moment it happens, IDF1 asks for the one
pairing of full trajectories that explains the most frames. Maximising IDTP over a one-to-one
track pairing is equivalent to minimising IDFN + IDFP, because IDFN = |gt| - IDTP and
IDFP = |pred| - IDTP for a fixed pairing.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from eval.common import IOU_MATCH, Sequence, iou_matrix
from tracker.assignment import linear_sum_assignment


@dataclass(frozen=True)
class IdResult:
    idf1: float
    idp: float
    idr: float
    idtp: int
    idfn: int
    idfp: int


def idf1(gt: Sequence, pred: Sequence, iou_thr: float = IOU_MATCH) -> IdResult:
    all_g = np.concatenate([f.ids for f in gt]) if gt else np.empty(0, dtype=np.int64)
    all_p = np.concatenate([f.ids for f in pred]) if pred else np.empty(0, dtype=np.int64)
    gt_uniq, gt_cnt = np.unique(all_g, return_counts=True)
    pr_uniq, pr_cnt = np.unique(all_p, return_counts=True)
    co = np.zeros((len(gt_uniq), len(pr_uniq)))  # frames where both present and IoU >= thr
    for g, p in zip(gt, pred):
        if len(g.ids) == 0 or len(p.ids) == 0:
            continue
        hit = iou_matrix(g.boxes, p.boxes) >= iou_thr - 1e-9
        gi = np.searchsorted(gt_uniq, g.ids)
        pi = np.searchsorted(pr_uniq, p.ids)
        co[np.ix_(gi, pi)] += hit
    idtp = 0
    if co.size:
        r, c = linear_sum_assignment(co, maximize=True)
        idtp = int(co[r, c].sum())
    n_g, n_p = int(gt_cnt.sum()), int(pr_cnt.sum())
    idfn, idfp = n_g - idtp, n_p - idtp
    idp = idtp / n_p if n_p else 0.0
    idr = idtp / n_g if n_g else 0.0
    f1 = 2 * idtp / (n_g + n_p) if (n_g + n_p) else 0.0
    return IdResult(f1, idp, idr, idtp, idfn, idfp)
