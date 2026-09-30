"""Appearance memory: each track keeps a short history of embeddings, not just the latest.

Matching uses the *minimum* cosine distance over the gallery, so one good view (e.g. the
front of a person seen before they walked behind a pillar) is enough.

Two update policies, because this is the single most consequential decision in the repo:

  always  add the embedding of every matched detection. Naive, and poisonable: when a person
          is partly hidden behind someone else, the detector box still contains the occluder's
          pixels, the embedding looks like the occluder, and the track starts remembering the
          wrong person. After the occlusion it happily matches the occluder instead.
  gated   add only when the detection is confident AND unoccluded (its box barely overlaps any
          other detection). During an occlusion the gallery is therefore frozen at what the
          object looked like before it disappeared.
"""
from __future__ import annotations

from collections import deque

import numpy as np


class Gallery:
    def __init__(self, size: int = 30, policy: str = "gated", update_conf: float = 0.6,
                 occ_iou: float = 0.3) -> None:
        if policy not in ("always", "gated"):
            raise ValueError(f"unknown gallery policy {policy!r}")
        self.feats: deque[np.ndarray] = deque(maxlen=size)
        self.policy, self.update_conf, self.occ_iou = policy, update_conf, occ_iou

    def __len__(self) -> int:
        return len(self.feats)

    def seed(self, feat: np.ndarray) -> None:
        """First embedding of a track: always stored, whatever the policy."""
        self.feats.append(feat)

    def should_update(self, score: float, max_overlap: float) -> bool:
        if self.policy == "always":
            return True
        return score >= self.update_conf and max_overlap < self.occ_iou

    def update(self, feat: np.ndarray, score: float, max_overlap: float) -> bool:
        if self.should_update(score, max_overlap):
            self.feats.append(feat)
            return True
        return False

    def distance(self, feats: np.ndarray) -> np.ndarray:
        """Min cosine distance from each row of `feats` (N, D, unit norm) to the gallery -> (N,)."""
        if not self.feats:
            return np.full(len(feats), 2.0)
        sims = feats @ np.stack(self.feats).T  # (N, G)
        return 1.0 - sims.max(axis=1)
