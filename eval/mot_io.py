"""Read MOTChallenge files. Frames are 1-indexed on disk, 0-indexed in memory.

gt.txt columns: frame, id, x, y, w, h, flag, class, visibility
We keep class 1 (pedestrian) with flag 1, the official MOT17 evaluation set. Distractor
classes (sitting people, reflections, ...) are NOT used to forgive false positives as the
official toolkit does, so absolute numbers are slightly pessimistic vs the leaderboard.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from eval.common import Sequence, sequence_from_tracks


def seq_length(seq_dir: Path) -> int:
    for line in (seq_dir / "seqinfo.ini").read_text().splitlines():
        if line.lower().startswith("seqlength"):
            return int(line.split("=")[1])
    raise ValueError(f"no seqLength in {seq_dir}/seqinfo.ini")


def load_mot_gt(seq_dir: Path) -> tuple[Sequence, dict[tuple[int, int], float]]:
    """Return (GT sequence, visibility[(gt_id, frame)])."""
    n = seq_length(seq_dir)
    rows = np.loadtxt(seq_dir / "gt" / "gt.txt", delimiter=",", ndmin=2)
    rows = rows[(rows[:, 6] == 1) & (rows[:, 7] == 1)]
    tracks: dict[int, dict[int, tuple[float, float, float, float]]] = {}
    vis: dict[tuple[int, int], float] = {}
    for f, i, x, y, w, h, _, _, v in rows:
        f0, i = int(f) - 1, int(i)
        tracks.setdefault(i, {})[f0] = (x, y, x + w, y + h)
        vis[(i, f0)] = float(v)
    return sequence_from_tracks(tracks, n), vis


def hidden_frames(vis: dict[tuple[int, int], float], min_visibility: float) -> set[tuple[int, int]]:
    """(gt_id, frame) pairs to treat as occluded when looking for occlusion gaps."""
    return {k for k, v in vis.items() if v < min_visibility}
