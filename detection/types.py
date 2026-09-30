"""What a detector hands to a tracker. Trackers never see images, only this."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Detections:
    boxes: np.ndarray  # (N, 4) xyxy
    scores: np.ndarray  # (N,)
    feats: np.ndarray | None = None  # (N, D) L2-normalised ReID embeddings, if extracted

    def __len__(self) -> int:
        return len(self.boxes)


def empty_detections() -> Detections:
    return Detections(np.empty((0, 4)), np.empty(0))
