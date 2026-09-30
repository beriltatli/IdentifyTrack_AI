"""HOTA (Luiten et al. 2021) with the DetA / AssA decomposition kept explicit.

    DetA_a = TP / (TP + FN + FP)                         -- detection only, IDs never used
    A(c)   = TPA(c) / (TPA(c) + FNA(c) + FPA(c))         -- per (gt track, pred track) pair
    AssA_a = (1/TP) * sum_{c in TP} A(c)                 -- association only
    HOTA_a = sqrt(DetA_a * AssA_a),  then averaged over alpha in {0.05, ..., 0.95}

Per-frame matching at threshold alpha maximises (global track alignment) x (IoU) so that
ties between equally-overlapping detections are broken in favour of the pairing that is
consistent over the whole sequence, rather than arbitrarily.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from eval.common import Sequence, iou_matrix
from tracker.assignment import linear_sum_assignment

ALPHAS = np.round(np.arange(1, 20) * 0.05, 2)  # 0.05 .. 0.95 (rounded: 12*0.05 != 0.6 in floats)
_EPS = 1e-9


@dataclass(frozen=True)
class HotaResult:
    hota: float
    deta: float
    assa: float
    hota_per_alpha: np.ndarray
    deta_per_alpha: np.ndarray
    assa_per_alpha: np.ndarray
    tp: np.ndarray  # per alpha
    fn: np.ndarray
    fp: np.ndarray


def _index_ids(seq: Sequence) -> tuple[np.ndarray, np.ndarray]:
    """Return (sorted unique ids, per-id frame count)."""
    all_ids = np.concatenate([f.ids for f in seq]) if seq else np.empty(0, dtype=np.int64)
    uniq, counts = np.unique(all_ids, return_counts=True)
    return uniq, counts


def hota(gt: Sequence, pred: Sequence, alphas: np.ndarray = ALPHAS) -> HotaResult:
    if len(gt) != len(pred):
        raise ValueError("gt and pred must cover the same number of frames")
    gt_uniq, gt_cnt = _index_ids(gt)
    pr_uniq, pr_cnt = _index_ids(pred)
    n_g, n_p = len(gt_uniq), len(pr_uniq)
    total_gt, total_pr = int(gt_cnt.sum()), int(pr_cnt.sum())

    # Per-frame similarity + integer id indices, computed once.
    frames = []
    overlap_sum = np.zeros((n_g, n_p))
    for g, p in zip(gt, pred):
        gi = np.searchsorted(gt_uniq, g.ids)
        pi = np.searchsorted(pr_uniq, p.ids)
        sim = iou_matrix(g.boxes, p.boxes)
        frames.append((gi, pi, sim))
        if sim.size:
            overlap_sum[np.ix_(gi, pi)] += sim
    # Global alignment: soft Jaccard of the two tracks' lifetimes, in [0, 1].
    denom = gt_cnt[:, None] + pr_cnt[None, :] - overlap_sum
    align = np.divide(overlap_sum, denom, out=np.zeros_like(overlap_sum), where=denom > _EPS)

    n_a = len(alphas)
    tp = np.zeros(n_a)
    assa = np.zeros(n_a)
    for ai, alpha in enumerate(alphas):
        match_cnt = np.zeros((n_g, n_p))
        for gi, pi, sim in frames:
            if sim.size == 0:
                continue
            ok = sim >= alpha - _EPS
            score = align[np.ix_(gi, pi)] * sim
            cost = np.where(ok, -score, np.inf)
            r, c = linear_sum_assignment(cost)
            match_cnt[gi[r], pi[c]] += 1
        tp[ai] = match_cnt.sum()
        if tp[ai] > 0:
            a_den = gt_cnt[:, None] + pr_cnt[None, :] - match_cnt  # TPA + FNA + FPA
            a_c = np.divide(match_cnt, a_den, out=np.zeros_like(match_cnt), where=a_den > 0)
            assa[ai] = float((match_cnt * a_c).sum() / tp[ai])

    fn = total_gt - tp
    fp = total_pr - tp
    det_den = tp + fn + fp
    deta = np.divide(tp, det_den, out=np.zeros(n_a), where=det_den > 0)
    hota_a = np.sqrt(deta * assa)
    return HotaResult(
        hota=float(hota_a.mean()),
        deta=float(deta.mean()),
        assa=float(assa.mean()),
        hota_per_alpha=hota_a,
        deta_per_alpha=deta,
        assa_per_alpha=assa,
        tp=tp,
        fn=fn,
        fp=fp,
    )
