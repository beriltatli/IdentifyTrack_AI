"""Shared plumbing for track / evaluate / sweep / reproduce: config, data prep, running a
method on prepared detections, and scoring a pooled set of sequences."""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from baselines.appearance_only import run_appearance_only
from baselines.greedy_iou import run_greedy_iou
from baselines.kalman_iou import run_kalman_iou
from detection.frozen_detector import (YoloDetector, load_dets, load_mot17_public, save_dets)
from detection.reid import attach_features, make_extractor
from detection.types import Detections
from eval.clear import clear_match
from eval.common import Sequence, concat_sequences
from eval.gaps import fragmentation, gap_recovery
from eval.hota import hota
from eval.idf1 import idf1
from eval.mot_io import hidden_frames, load_mot_gt, seq_length
from eval.oracle import oracle_detections
from eval.report import RunMetrics
from eval.taxonomy import classify_switches
from tracker.tracker import Tracker, TrackerConfig

CACHE = Path("cache")
ID_STRIDE = 1_000_000


def load_config(path: str = "config.yaml") -> dict:
    return yaml.safe_load(Path(path).read_text())


def high_only(dets: list[Detections], thr: float) -> list[Detections]:
    """Baselines see only detections >= output_conf (the emission contract)."""
    out = []
    for d in dets:
        k = d.scores >= thr
        out.append(Detections(d.boxes[k], d.scores[k], None if d.feats is None else d.feats[k]))
    return out


def prepare(cfg: dict, seq: str, *, oracle: bool = False) -> tuple[list[Detections], float]:
    """Detections (+ReID features) for one sequence, cached. Returns (dets, detector seconds)."""
    root = Path(cfg["data"]["root"]) / seq
    d, r = cfg["detector"], cfg["reid"]
    tag = "oracle" if oracle else d["backend"]
    path = CACHE / f"{tag}_{r['backend']}" / f"{seq}.npz"
    if path.exists():
        return load_dets(path), 0.0
    t0 = time.time()
    if oracle:
        gt, vis = load_mot_gt(root)
        dets = oracle_detections(gt, hidden_frames(vis, cfg["data"]["min_visibility"]))
    elif d["backend"] == "mot17_public":
        dets = load_mot17_public(root, d["conf_floor"])
    elif d["backend"] == "yolo":
        det = YoloDetector(d["yolo_weights"], d["yolo_imgsz"], d["conf_floor"], d["person_class"])
        dets = det([root / "img1" / f"{i:06d}.jpg" for i in range(1, seq_length(root) + 1)])
    else:
        raise ValueError(d["backend"])
    det_seconds = time.time() - t0
    dets = attach_features(root, dets, make_extractor(r))
    save_dets(dets, path)
    return dets, det_seconds


def run_method(name: str, cfg: dict, dets: list[Detections], overrides: dict | None = None):
    """Run one tracking method over one sequence. Returns (Sequence, extra info dict)."""
    oc = cfg["output_conf"]
    if name == "greedy_iou":
        return run_greedy_iou(high_only(dets, oc), **cfg["baselines"]["greedy_iou"]), {}
    if name == "kalman_iou":
        return run_kalman_iou(high_only(dets, oc), **cfg["baselines"]["kalman_iou"]), {}
    if name == "appearance_only":
        return run_appearance_only(dets, output_conf=oc, **cfg["baselines"]["appearance_only"]), {}
    if name == "ours":
        tc = {**cfg["tracker"], "output_conf": oc, **(overrides or {})}
        trk = Tracker(TrackerConfig.from_dict(tc))
        seq = [trk.update(d) for d in dets]
        return seq, {"reacquisitions": trk.reacquisitions}
    raise ValueError(name)


@dataclass
class Pool:
    """Ground truth of all evaluation sequences concatenated (see eval.common.concat_sequences)."""
    seqs: list[str]
    gt: Sequence
    hidden: set[tuple[int, int]]
    offsets: list[int]


def load_pool(cfg: dict) -> Pool:
    gts, hidden, offsets, off = [], set(), [], 0
    for k, s in enumerate(cfg["data"]["sequences"]):
        gt, vis = load_mot_gt(Path(cfg["data"]["root"]) / s)
        h = hidden_frames(vis, cfg["data"]["min_visibility"])
        hidden |= {(g + k * ID_STRIDE, f + off) for g, f in h}
        gts.append(gt)
        offsets.append(off)
        off += len(gt)
    return Pool(list(cfg["data"]["sequences"]), concat_sequences(gts), hidden, offsets)


def score(name: str, pool: Pool, preds: list[Sequence], cfg: dict) -> RunMetrics:
    pred = concat_sequences(preds)
    c = clear_match(pool.gt, pred, cfg["eval"]["iou_match"])
    return RunMetrics(
        name, hota(pool.gt, pred), idf1(pool.gt, pred, cfg["eval"]["iou_match"]), c,
        gap_recovery(pool.gt, c, pool.hidden), fragmentation(pool.gt, c),
        classify_switches(pool.gt, c, cfg["eval"]["swap_window"], pool.hidden),
    )


def write_mot(seq: Sequence, path: Path) -> None:
    """Results in MOTChallenge text format (frame,id,x,y,w,h,1,-1,-1,-1)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for t, fr in enumerate(seq):
            for i, b in zip(fr.ids, fr.boxes):
                f.write(f"{t + 1},{int(i)},{b[0]:.2f},{b[1]:.2f},{b[2]-b[0]:.2f},{b[3]-b[1]:.2f},1,-1,-1,-1\n")


def metrics_row(r: RunMetrics, fps: float | None = None) -> dict:
    """Flat, JSON-safe summary of one run (numbers in the README come from these)."""
    lg = r.gaps.bins[-1]
    return {
        "name": r.name, "HOTA": 100 * r.hota.hota, "DetA": 100 * r.hota.deta, "AssA": 100 * r.hota.assa,
        "IDF1": 100 * r.idf1.idf1, "MOTA": 100 * r.clear.mota, "IDSW": r.clear.idsw,
        "long_gap_recovered": lg.recovered, "long_gap_n": lg.n,
        "long_gap_rate": None if lg.rate is None else 100 * lg.rate,
        "gap_bins": {b.label: [b.recovered, b.n] for b in r.gaps.bins},
        "taxonomy": dict(r.taxonomy.counts), "swap_incidents": r.taxonomy.swap_incidents,
        "frag_median": r.frag.median, "frag_p90": r.frag.p90, "fps": fps,
    }
