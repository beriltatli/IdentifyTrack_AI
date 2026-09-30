"""Occlusion-gap stratified identity recovery, long-gap rate, and fragmentation.

A *gap* is a maximal run of frames in which a GT track has no annotation (occluded / out of
view) between two frames where it does. For each reappearance we ask: does the object get
back the pred ID it had before it vanished?

Definitions (deliberately strict, and fixed here so they can't drift to flatter a result):
  * "before" ID  = the pred ID of the GT track's last match at or before the gap start.
                   No such match -> the reappearance is excluded (nothing to give back).
  * "after"  ID  = the first match within the contiguous visible segment after the gap.
                   No match in that segment -> counted as NOT recovered (`never_reacquired`).
  * recovered    = after == before.
  * Bins are [1,5], [6,15], [16,39], [40,inf) so that "40+" literally means >= 40 frames.
A gap here can also be an object that walked out of frame and back; GT alone can't tell
those apart. Sequences with GT visibility flags should filter upstream.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from eval.clear import ClearResult
from eval.common import Sequence

BINS: list[tuple[str, int, float]] = [
    ("1-5", 1, 5),
    ("6-15", 6, 15),
    ("16-39", 16, 39),
    ("40+", 40, float("inf")),
]


@dataclass(frozen=True)
class GapEvent:
    gt_id: int
    gap_start: int  # first absent frame
    gap_len: int
    recovered: bool
    never_reacquired: bool


@dataclass(frozen=True)
class BinStats:
    label: str
    n: int
    recovered: int

    @property
    def rate(self) -> float | None:
        return self.recovered / self.n if self.n else None


@dataclass(frozen=True)
class GapResult:
    events: list[GapEvent]
    bins: list[BinStats]
    excluded_no_prior_id: int
    long_gap_recovery: float | None  # the "40+" bin, promoted


def _gt_presence(
    gt: Sequence, hidden: set[tuple[int, int]] | None = None
) -> dict[int, list[int]]:
    """Frames where each GT track counts as visible. `hidden` = (gt_id, frame) pairs to treat
    as absent (e.g. annotated but visibility < threshold): that is what turns MOT17's
    "still annotated while behind a pillar" into an occlusion gap."""
    frames: dict[int, list[int]] = {}
    for t, f in enumerate(gt):
        for i in f.ids:
            if hidden and (int(i), t) in hidden:
                continue
            frames.setdefault(int(i), []).append(t)
    return frames


def gap_recovery(
    gt: Sequence, clear: ClearResult, hidden: set[tuple[int, int]] | None = None
) -> GapResult:
    pred_at: dict[int, dict[int, int]] = {}  # gt_id -> {frame: pred_id}
    for t, pairs in enumerate(clear.matches):
        for g, p in pairs:
            pred_at.setdefault(g, {})[t] = p
    events: list[GapEvent] = []
    excluded = 0
    for gid, present in _gt_presence(gt, hidden).items():
        matched = pred_at.get(gid, {})
        matched_frames = sorted(matched)
        for k in range(len(present) - 1):
            a, b = present[k], present[k + 1]
            if b - a == 1:
                continue
            before_frames = [f for f in matched_frames if f <= a]
            if not before_frames:
                excluded += 1
                continue
            before = matched[before_frames[-1]]
            # end of the contiguous visible segment starting at b
            end = b
            j = k + 1
            while j + 1 < len(present) and present[j + 1] == present[j] + 1:
                j += 1
            end = present[j]
            after_frames = [f for f in matched_frames if b <= f <= end]
            if not after_frames:
                events.append(GapEvent(gid, a + 1, b - a - 1, False, True))
            else:
                events.append(
                    GapEvent(gid, a + 1, b - a - 1, matched[after_frames[0]] == before, False)
                )
    bins = []
    for label, lo, hi in BINS:
        sel = [e for e in events if lo <= e.gap_len <= hi]
        bins.append(BinStats(label, len(sel), sum(e.recovered for e in sel)))
    return GapResult(events, bins, excluded, bins[-1].rate)


@dataclass(frozen=True)
class FragmentationResult:
    per_track: dict[int, int]  # gt_id -> number of distinct pred IDs it was matched to
    median: float
    p90: float
    unmatched_tracks: int  # GT tracks never matched at all (excluded from the distribution)


def summarize_fragmentation(counts: list[int]) -> tuple[float, float]:
    if not counts:
        return 0.0, 0.0
    return float(np.median(counts)), float(np.percentile(counts, 90))


def fragmentation(gt: Sequence, clear: ClearResult) -> FragmentationResult:
    ids: dict[int, set[int]] = {}
    for pairs in clear.matches:
        for g, p in pairs:
            ids.setdefault(g, set()).add(p)
    all_gt = set(_gt_presence(gt))
    per_track = {g: len(s) for g, s in ids.items()}
    med, p90 = summarize_fragmentation(list(per_track.values()))
    return FragmentationResult(per_track, med, p90, len(all_gt - set(ids)))
