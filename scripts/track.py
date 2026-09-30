"""Run one method on the configured sequences and write MOTChallenge result files.

    python -m scripts.track --method ours --out results/tracks
"""
from __future__ import annotations

import argparse
from pathlib import Path

from scripts.runlib import load_config, prepare, run_method, write_mot

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="ours", choices=["greedy_iou", "kalman_iou", "appearance_only", "ours"])
    ap.add_argument("--out", default="results/tracks")
    ap.add_argument("--oracle", action="store_true")
    a = ap.parse_args()
    cfg = load_config()
    for s in cfg["data"]["sequences"]:
        seq, _ = run_method(a.method, cfg, prepare(cfg, s, oracle=a.oracle)[0])
        write_mot(seq, Path(a.out) / a.method / f"{s}.txt")
        print("wrote", Path(a.out) / a.method / f"{s}.txt")
