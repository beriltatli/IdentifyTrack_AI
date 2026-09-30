"""Track state machine: tentative -> confirmed -> lost -> deleted.

  tentative  born from an unmatched detection; deleted on its first miss (likely a false start)
             and confirmed after `n_init` matches.
  confirmed  trusted; one miss sends it to `lost`.
  lost       keeps coasting on the Kalman prediction and stays eligible for re-matching. It is
             deleted once it has gone `lost_buffer` consecutive frames unmatched. This buffer
             is the direct lever on long-gap recovery: a gap longer than the buffer cannot be
             recovered by construction.
  lost -> confirmed on a match: that is the re-identification event.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from tracker.gallery import Gallery
from tracker.kalman import KalmanBoxFilter


class State(Enum):
    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    LOST = "lost"
    DELETED = "deleted"


@dataclass
class Track:
    id: int
    kf: KalmanBoxFilter
    gallery: Gallery
    state: State = State.TENTATIVE
    hits: int = 1
    misses: int = 0  # consecutive frames unmatched (time since update)
    born: int = 0

    @property
    def alive(self) -> bool:
        return self.state is not State.DELETED

    def on_match(self, n_init: int) -> bool:
        """Register a match. Returns True if this was a lost track being re-acquired."""
        reacquired = self.state is State.LOST
        self.hits += 1
        self.misses = 0
        if self.state is State.LOST or (self.state is State.TENTATIVE and self.hits >= n_init):
            self.state = State.CONFIRMED
        return reacquired

    def on_miss(self, lost_buffer: int) -> None:
        self.misses += 1
        if self.state is State.TENTATIVE:
            self.state = State.DELETED
        elif self.state is State.CONFIRMED:
            self.state = State.LOST
        if self.state is State.LOST and self.misses > lost_buffer:
            self.state = State.DELETED
