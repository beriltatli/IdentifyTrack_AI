"""Hand-built synthetic tracks. Boxes are 10x10 squares placed far apart unless stated."""
from __future__ import annotations

from eval.common import Frame, Sequence, sequence_from_tracks


def box(x: float, y: float = 0.0, w: float = 10.0, h: float = 10.0):
    return (x, y, x + w, y + h)


def static_track(x: float, frames: range | list[int], y: float = 0.0):
    return {f: box(x, y) for f in frames}


def build(tracks, n: int) -> Sequence:
    return sequence_from_tracks(tracks, n)
