"""Run methods on the configured sequences and print every table.

    python -m scripts.evaluate                      # all four methods, real detections
    python -m scripts.evaluate --oracle             # GT detections (association-only)
    python -m scripts.evaluate --methods kalman_iou ours
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from eval.report import gap_table, main_table, taxonomy_table
from scripts.runlib import (Pool, load_config, load_pool, prepare, run_method, score, write_mot)

ALL = ["greedy_iou", "kalman_iou", "appearance_only", "ours"]


def evaluate(cfg: dict, methods: list[str], oracle: bool = False, overrides: dict | None = None,
             save_dir: Path | None = None, pool: Pool | None = None):
    pool = pool or load_pool(cfg)
    prepared = {s: prepare(cfg, s, oracle=oracle)[0] for s in pool.seqs}
    runs, fps = [], {}
    for m in methods:
        t0 = time.time()
        preds = []
        for s in pool.seqs:
            seq, _ = run_method(m, cfg, prepared[s], overrides if m == "ours" else None)
            preds.append(seq)
            if save_dir:
                write_mot(seq, save_dir / m / f"{s}.txt")
        frames = sum(len(p) for p in preds)
        fps[m] = frames / (time.time() - t0)
        runs.append(score(m, pool, preds, cfg))
    return runs, fps


def print_report(runs, fps, title: str) -> None:
    print(f"\n=== {title} ===\n")
    print(main_table(runs)); print()
    print("Occlusion-gap recovery (recovered / reappearances):"); print(gap_table(runs)); print()
    print("ID-switch taxonomy (events):"); print(taxonomy_table(runs)); print()
    for r in runs:
        print(f"{r.name:<16} fragmentation per GT track: median={r.frag.median:.1f} p90={r.frag.p90:.1f}"
              f" | tracker-only FPS={fps[r.name]:.0f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=ALL)
    ap.add_argument("--oracle", action="store_true")
    ap.add_argument("--save", default=None)
    ap.add_argument("--reid", default=None, help="override reid backend; app_gate is re-calibrated for it")
    a = ap.parse_args()
    cfg = load_config()
    if a.reid and a.reid != cfg["reid"]["backend"]:
        from scripts.calibrate import calibrate
        cfg["reid"] = dict(cfg["reid"], backend=a.reid)
        gate = round(calibrate(cfg)["app_gate"], 2)
        cfg["tracker"]["app_gate"] = gate
        cfg["baselines"]["appearance_only"]["app_gate"] = gate
        print(f"[{a.reid}] app_gate re-calibrated to {gate} (same-identity p99 cosine distance)")
    runs, fps = evaluate(cfg, a.methods, a.oracle, save_dir=Path(a.save) if a.save else None)
    print_report(runs, fps, ("ORACLE (GT) detections" if a.oracle else
                             f"detector={cfg['detector']['backend']}") + f" | reid={cfg['reid']['backend']}")
