"""The tracker loop: Kalman motion + gated appearance gallery + lifecycle + two-stage matching.

Per frame:
  0. every live track predicts (lost tracks keep coasting).
  1. confirmed+lost tracks  vs HIGH-confidence detections: gated motion/appearance cost, Hungarian.
  1b. tentative tracks      vs the still-unmatched high detections: IoU only.
  2. still-unmatched confirmed+lost tracks vs LOW-confidence detections: IoU only. A weak box
     in the middle of an occlusion is exactly the evidence that keeps a track alive. Low boxes
     update the motion state but are never emitted and never enter the gallery (under the
     gated policy).
  3. unmatched high detections spawn tentative tracks; unmatched tracks age / die.

Emission contract (shared with every baseline, see baselines/): the output is exactly the
detections with score >= output_conf, raw boxes, each carrying a track id. Tracking decides
ids, never which boxes exist, so DetA cannot move between variants.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields

import numpy as np

from detection.types import Detections
from eval.common import Frame, empty_frame, iou_matrix
from tracker.assignment import linear_sum_assignment
from tracker.cost import fuse_cost
from tracker.gallery import Gallery
from tracker.kalman import KalmanBoxFilter
from tracker.lifecycle import State, Track


@dataclass
class TrackerConfig:
    output_conf: float = 0.30  # high/low split AND the emission threshold
    n_init: int = 3
    lost_buffer: int = 100
    motion_gate: float = 0.0  # active tracks: squared-Mahalanobis gate, <=0 = IoU only (see config.yaml)
    lost_motion_gate: float = 9.4877  # chi-square 95%: a coasting track's covariance is already inflated
    motion_iou_gate: float = 0.3  # ...or IoU with the predicted box, whichever accepts the pair
    app_gate: float = 0.5  # max cosine distance for a pair to be possible at all
    lambda_motion: float = 0.5  # weight of motion vs appearance among feasible pairs
    tentative_iou: float = 0.3
    stage2_iou: float = 0.3
    use_appearance: bool = True
    gallery_size: int = 30
    gallery_policy: str = "gated"  # "gated" | "always"
    gallery_update_conf: float = 0.6
    gallery_occ_iou: float = 0.3

    @classmethod
    def from_dict(cls, d: dict) -> "TrackerConfig":
        known = {f.name for f in fields(cls)}
        bad = set(d) - known
        if bad:
            raise KeyError(f"unknown tracker config keys: {sorted(bad)}")
        return cls(**d)


class Tracker:
    def __init__(self, cfg: TrackerConfig | None = None) -> None:
        self.cfg = cfg or TrackerConfig()
        self.tracks: list[Track] = []
        self._next_id = 1
        self.frame = 0
        self.reacquisitions: list[tuple[int, int, int]] = []  # (frame, track id, frames lost)

    # ------------------------------------------------------------------ helpers
    def _new_gallery(self) -> Gallery:
        c = self.cfg
        return Gallery(c.gallery_size, c.gallery_policy, c.gallery_update_conf, c.gallery_occ_iou)

    def _spawn(self, det_box, feat) -> Track:
        tr = Track(self._next_id, KalmanBoxFilter(det_box), self._new_gallery(), born=self.frame)
        self._next_id += 1
        if feat is not None:
            tr.gallery.seed(feat)
        self.tracks.append(tr)
        return tr

    def _apply_match(self, tr: Track, box, feat, score, overlap, *, allow_gallery: bool) -> None:
        tr.kf.update(box)
        lost_for = tr.misses
        if tr.on_match(self.cfg.n_init) and lost_for:
            self.reacquisitions.append((self.frame, tr.id, lost_for))
        if feat is not None and allow_gallery:
            tr.gallery.update(feat, score, overlap)

    @staticmethod
    def _overlaps(boxes: np.ndarray) -> np.ndarray:
        """Max IoU of each detection with any *other* detection in the frame (occlusion proxy)."""
        if len(boxes) < 2:
            return np.zeros(len(boxes))
        iou = iou_matrix(boxes, boxes)
        np.fill_diagonal(iou, 0.0)
        return iou.max(axis=1)

    # ------------------------------------------------------------------ main step
    def update(self, dets: Detections) -> Frame:
        c = self.cfg
        feats = dets.feats if c.use_appearance else None
        high = np.flatnonzero(dets.scores >= c.output_conf)
        low = np.flatnonzero(dets.scores < c.output_conf)
        overlap = self._overlaps(dets.boxes)
        preds = {tr.id: tr.kf.predict() for tr in self.tracks}
        ids = np.zeros(len(dets), dtype=np.int64)
        matched: set[int] = set()  # track ids matched this frame

        def feat_of(j):
            return None if feats is None else feats[j]

        # ---- stage 1: high detections vs confirmed tracks, THEN lost tracks on what is left.
        # A cascade by recency, as in DeepSORT: a long-coasting track has an inflated covariance,
        # which makes its Mahalanobis distance small to *everything*, so in a joint assignment it
        # would steal detections from the track that is actually following the person.
        cand = [t for t in self.tracks if t.state in (State.CONFIRMED, State.LOST)]
        for group_state in (State.CONFIRMED, State.LOST):
            group = [t for t in cand if t.state is group_state and t.id not in matched]
            avail = [int(j) for j in high if ids[j] == 0]
            if not group or not avail:
                continue
            hb = dets.boxes[avail]
            maha = np.stack([t.kf.mahalanobis_sq(hb) for t in group])
            app = None
            if feats is not None:
                app = np.stack([t.gallery.distance(feats[avail]) for t in group])
            iou = iou_matrix(np.array([t.kf.box for t in group]), hb)
            gate = c.motion_gate if group_state is State.CONFIRMED else c.lost_motion_gate
            cost = fuse_cost(maha, app, gate, c.app_gate, c.lambda_motion, iou, c.motion_iou_gate)
            for ti, hj in zip(*linear_sum_assignment(cost)):
                tr, j = group[ti], avail[hj]
                self._apply_match(tr, dets.boxes[j], feat_of(j), dets.scores[j], overlap[j],
                                  allow_gallery=True)
                ids[j] = tr.id
                matched.add(tr.id)

        # ---- stage 1b: tentative vs leftover high detections (IoU)
        left_high = [int(j) for j in high if ids[j] == 0]
        tent = [t for t in self.tracks if t.state is State.TENTATIVE]
        if tent and left_high:
            iou = iou_matrix(np.array([preds[t.id] for t in tent]), dets.boxes[left_high])
            cost = np.where(iou >= c.tentative_iou, -iou, np.inf)
            for ti, hj in zip(*linear_sum_assignment(cost)):
                tr, j = tent[ti], left_high[hj]
                self._apply_match(tr, dets.boxes[j], feat_of(j), dets.scores[j], overlap[j],
                                  allow_gallery=True)
                ids[j] = tr.id
                matched.add(tr.id)

        # ---- stage 2: leftover confirmed/lost vs LOW detections (IoU, motion only)
        rest = [t for t in cand if t.id not in matched]
        if rest and len(low):
            iou = iou_matrix(np.array([preds[t.id] for t in rest]), dets.boxes[low])
            cost = np.where(iou >= c.stage2_iou, -iou, np.inf)
            for ti, lj in zip(*linear_sum_assignment(cost)):
                tr, j = rest[ti], int(low[lj])
                # naive "always" policy lets even these weak, possibly occluded boxes in
                self._apply_match(tr, dets.boxes[j], feat_of(j), dets.scores[j], overlap[j],
                                  allow_gallery=c.gallery_policy == "always")
                matched.add(tr.id)  # ids[j] stays 0: low boxes are not emitted

        # ---- unmatched high detections spawn tentative tracks (and are emitted)
        for j in high:
            if ids[j] == 0:
                ids[j] = self._spawn(dets.boxes[j], feat_of(j)).id
                matched.add(int(ids[j]))
        for tr in self.tracks:
            if tr.id not in matched:
                tr.on_miss(c.lost_buffer)
        self.tracks = [t for t in self.tracks if t.alive]
        self.frame += 1

        if not len(high):
            return empty_frame()
        return Frame(ids[high].copy(), dets.boxes[high].copy())


def run_tracker(dets: list[Detections], cfg: TrackerConfig | None = None):
    """Convenience: run a whole sequence. Returns (Sequence, Tracker) so callers can inspect
    `tracker.reacquisitions`."""
    trk = Tracker(cfg)
    return [trk.update(d) for d in dets], trk
