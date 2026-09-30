"""Shared data types for the evaluation suite.

A sequence is a list indexed by frame number; each entry holds the IDs and xyxy boxes
present in that frame. Ground truth and tracker output use the same type, which is what
lets the oracle run (tracker on GT detections) reuse every metric unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

IOU_MATCH = 0.5  # threshold for CLEAR/IDF1/gap analysis (HOTA sweeps its own alphas)


@dataclass(frozen=True)
class Frame:
    ids: np.ndarray  # (N,) int64
    boxes: np.ndarray  # (N, 4) float64, xyxy


Sequence = list[Frame]


def empty_frame() -> Frame:
    return Frame(np.empty(0, dtype=np.int64), np.empty((0, 4), dtype=np.float64))


def sequence_from_tracks(
    tracks: dict[int, dict[int, tuple[float, float, float, float]]], num_frames: int
) -> Sequence:
    """Build a Sequence from {track_id: {frame: (x1,y1,x2,y2)}}. Used by tests and loaders."""
    per_frame: list[list[tuple[int, tuple[float, float, float, float]]]] = [
        [] for _ in range(num_frames)
    ]
    for tid, frames in tracks.items():
        for f, box in frames.items():
            per_frame[f].append((tid, box))
    seq: Sequence = []
    for items in per_frame:
        if not items:
            seq.append(empty_frame())
            continue
        items.sort(key=lambda x: x[0])
        seq.append(
            Frame(
                np.array([i for i, _ in items], dtype=np.int64),
                np.array([b for _, b in items], dtype=np.float64),
            )
        )
    return seq


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU of xyxy boxes, shape (len(a), len(b))."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.where(union > 0, union, 1.0), 0.0)


def concat_sequences(seqs: list[Sequence], id_stride: int = 1_000_000) -> Sequence:
    """Join sequences end to end with disjoint IDs so they can be scored as one pool.

    Matching is per frame and IDs never collide, so this equals summing TP/FN/FP counts
    over sequences (how HOTA is pooled) rather than averaging per-sequence scores.
    """
    out: Sequence = []
    for k, s in enumerate(seqs):
        out.extend(Frame(f.ids + k * id_stride, f.boxes) for f in s)
    return out
