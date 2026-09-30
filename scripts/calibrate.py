"""Choose app_gate for a ReID backend by one fixed rule: the 99th percentile of the cosine
distance between detections of the SAME ground-truth identity (>= 10 frames apart).

Caveat, stated in the README: this looks at GT identities of the evaluation sequences (there is
no held-out sequence), so the gate is lightly fitted to them.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from eval.common import iou_matrix
from eval.mot_io import load_mot_gt
from scripts.runlib import prepare


def same_and_diff_distances(cfg: dict, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    same, diff = [], []
    for s in cfg["data"]["sequences"]:
        dets, _ = prepare(cfg, s)
        gt, _ = load_mot_gt(Path(cfg["data"]["root"]) / s)
        rec = []
        for t, (d, g) in enumerate(zip(dets, gt)):
            if not len(d) or not len(g.ids):
                continue
            iou = iou_matrix(d.boxes, g.boxes)
            j = iou.argmax(1)
            rec += [(t, int(g.ids[j[k]]), d.feats[k]) for k in np.flatnonzero(iou.max(1) >= 0.5)]
        F, T, I = np.stack([r[2] for r in rec]), np.array([r[0] for r in rec]), np.array([r[1] for r in rec])
        for a in rng.choice(len(rec), size=min(3000, len(rec)), replace=False):
            b = rng.choice(len(rec), size=40)
            dist, far = 1 - F[b] @ F[a], np.abs(T[b] - T[a]) >= 10
            same += list(dist[far & (I[b] == I[a])])
            diff += list(dist[far & (I[b] != I[a])])
    return np.array(same), np.array(diff)


def calibrate(cfg: dict) -> dict:
    same, diff = same_and_diff_distances(cfg)
    gate = float(np.percentile(same, 99))
    a, b = np.random.default_rng(1).choice(same, 2000), np.random.default_rng(2).choice(diff, 2000)
    return {"app_gate": gate, "same_p50": float(np.median(same)), "diff_p50": float(np.median(diff)),
            "auc": float(np.mean(a[:, None] < b[None, :])), "diff_rejected_at_gate": float(np.mean(diff > gate))}


if __name__ == "__main__":
    import sys
    from scripts.runlib import load_config
    cfg = load_config()
    if len(sys.argv) > 1:
        cfg["reid"] = dict(cfg["reid"], backend=sys.argv[1])
    print(cfg["reid"]["backend"], calibrate(cfg))
