"""Cost matrix: gating first, then weighting.

A pair is *impossible* if its Mahalanobis motion distance exceeds the motion gate, or its
appearance distance exceeds the appearance gate. Impossible pairs get infinite cost and are
never traded against the other term. Only the survivors are scored:

    cost = lam * motion + (1 - lam) * (appearance / app_gate)      in [0, 1]

Why gate before weighting? A weighted sum lets a great appearance score pay for an absurd
motion jump (and vice versa), which is exactly how a track teleports onto a look-alike across
the scene. With gating, appearance can only choose between candidates that motion already
accepts (and motion between candidates appearance already accepts).
"""
from __future__ import annotations

import numpy as np


def fuse_cost(
    maha_sq: np.ndarray,
    app_dist: np.ndarray | None,
    motion_gate: float,
    app_gate: float,
    lam: float,
    iou: np.ndarray | None = None,
    iou_gate: float = 0.3,
) -> np.ndarray:
    """maha_sq, app_dist, iou: (tracks, dets). app_dist=None -> motion only.

    Motion evidence is two-sided because each signal fails somewhere the other does not:
    Mahalanobis breaks when a detector box is clipped at the image border or changes shape
    (IoU stays high), while IoU is useless for a track that has coasted for seconds and
    drifted (Mahalanobis widens with the growing covariance and still reaches it). A pair is
    motion-feasible if EITHER accepts it, and its motion cost is the better of the two.
    motion_gate <= 0 switches Mahalanobis off (IoU only). On MOT17 that is the better choice for
    tracks seen in the last frame: Mahalanobis admitted look-alike neighbours IoU rejects.
    """
    if motion_gate <= 0:  # Mahalanobis disabled: pure IoU motion evidence (as in SORT)
        if iou is None:
            raise ValueError("motion_gate <= 0 needs an IoU matrix")
        ok, motion = iou >= iou_gate, 1.0 - iou
    else:
        ok = maha_sq <= motion_gate
        motion = np.clip(maha_sq / motion_gate, 0.0, 1.0)
        if iou is not None:
            ok = ok | (iou >= iou_gate)
            motion = np.minimum(motion, 1.0 - iou)
    if app_dist is None:
        return np.where(ok, motion, np.inf)
    ok = ok & (app_dist <= app_gate)
    cost = lam * motion + (1.0 - lam) * (app_dist / app_gate)
    return np.where(ok, cost, np.inf)
