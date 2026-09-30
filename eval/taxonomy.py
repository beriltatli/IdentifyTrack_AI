"""Automatic classification of every identity switch by cause.

Each SwitchEvent (GT track g moves from pred old -> new at frame t) gets exactly one label,
checked in this order:

  swap         Mutual exchange: another GT track g' had `new` and simultaneously (within
               `swap_window` frames) receives `old`. Two events per incident. Matching failure.
  drift        `new` was last matched to a *different* GT object and there is no mutual
               exchange: an existing identity slid across to another object. Motion-model
               failure (the track's prediction/gate followed the wrong thing).
  gap-reassign The GT track was absent (annotated-out) between its last match and now, and
               `new` is not an ID taken from another object: it came back as a stranger.
               Memory failure.
  other        Continuously visible, `new` not taken from anyone: the old track died (lost
               buffer expired / detector miss) and a fresh ID started. NOT one of the three
               causes in the brief -- kept separate so the counts add up honestly instead of
               being forced into a category they do not belong to.

Order matters: an ID stolen from another object during/after an occlusion is a matching or
motion failure (someone else's identity was hijacked), not a memory failure.
"""
from __future__ import annotations

from dataclasses import dataclass

from eval.clear import ClearResult, SwitchEvent
from eval.common import Sequence

CAUSES = ("swap", "drift", "gap-reassign", "other")


@dataclass(frozen=True)
class ClassifiedSwitch:
    event: SwitchEvent
    cause: str
    donor_gt: int | None  # GT track that previously owned event.new_pred, if any


@dataclass(frozen=True)
class TaxonomyResult:
    switches: list[ClassifiedSwitch]
    counts: dict[str, int]  # events per cause (a swap incident contributes 2 events)
    swap_incidents: int

    @property
    def total(self) -> int:
        return len(self.switches)


def classify_switches(
    gt: Sequence,
    clear: ClearResult,
    swap_window: int = 5,
    hidden: set[tuple[int, int]] | None = None,
) -> TaxonomyResult:
    """`hidden` has the same meaning as in gaps.gap_recovery (frames treated as occluded)."""
    present: dict[int, set[int]] = {}
    for t, f in enumerate(gt):
        for i in f.ids:
            if hidden and (int(i), t) in hidden:
                continue
            present.setdefault(int(i), set()).add(t)

    # Replay matches to find who owned each pred ID just before each switch.
    by_frame: dict[int, list[SwitchEvent]] = {}
    for e in clear.switches:
        by_frame.setdefault(e.frame, []).append(e)
    owner: dict[int, int] = {}
    donor: dict[int, int | None] = {}  # id(event) -> donor gt
    for t, pairs in enumerate(clear.matches):
        for e in by_frame.get(t, []):
            d = owner.get(e.new_pred)
            donor[id(e)] = d if d != e.gt_id else None
        for g, p in pairs:
            owner[p] = g

    events = clear.switches

    def is_swap(e: SwitchEvent) -> bool:
        d = donor[id(e)]
        if d is None:
            return False
        return any(
            o.gt_id == d and o.old_pred == e.new_pred and o.new_pred == e.old_pred
            and abs(o.frame - e.frame) <= swap_window
            for o in events
        )

    out: list[ClassifiedSwitch] = []
    for e in events:
        d = donor[id(e)]
        if is_swap(e):
            cause = "swap"
        elif d is not None:
            cause = "drift"
        elif any(f not in present.get(e.gt_id, ()) for f in range(e.prev_frame + 1, e.frame)):
            cause = "gap-reassign"
        else:
            cause = "other"
        out.append(ClassifiedSwitch(e, cause, d))
    counts = {c: sum(1 for s in out if s.cause == c) for c in CAUSES}
    return TaxonomyResult(out, counts, counts["swap"] // 2)
