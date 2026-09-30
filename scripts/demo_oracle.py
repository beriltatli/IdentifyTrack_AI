"""Interim Phase-3 demo: baselines on ORACLE detections (GT boxes), real MOT17 ground truth.

    python -m scripts.demo_oracle              # all three sequences
    python -m scripts.demo_oracle MOT17-09     # one sequence, fastest

No detector, no torch. It answers: if detection were perfect (except that nobody can see
through an occluder), how well does each association strategy keep identities?
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

from baselines.greedy_iou import run_greedy_iou
from baselines.kalman_iou import run_kalman_iou
from eval.clear import clear_match
from eval.common import concat_sequences
from eval.gaps import fragmentation, gap_recovery
from eval.hota import hota
from eval.idf1 import idf1
from eval.mot_io import hidden_frames, load_mot_gt
from eval.oracle import oracle_detections
from eval.report import RunMetrics, gap_table, main_table, taxonomy_table
from eval.taxonomy import classify_switches

DATA = Path("data/MOT17")
MIN_VISIBILITY = 0.25  # locked before looking at any result (see config.yaml once it exists)
METHODS = {"greedy_iou": run_greedy_iou, "kalman_iou": run_kalman_iou}


def main(seqs: list[str]) -> None:
    gts, hiddens, dets = [], set(), []
    for k, s in enumerate(seqs):
        gt, vis = load_mot_gt(DATA / s)
        hidden = hidden_frames(vis, MIN_VISIBILITY)
        gts.append(gt)
        dets.append(oracle_detections(gt, hidden))
        stride = 1_000_000
        hiddens |= {(g + k * stride, f + sum(len(x) for x in gts[:-1])) for g, f in hidden}
    gt_all = concat_sequences(gts)
    runs = []
    for name, run in METHODS.items():
        t0 = time.time()
        pred = concat_sequences([run(d) for d in dets])
        c = clear_match(gt_all, pred)
        runs.append(RunMetrics(
            name, hota(gt_all, pred), idf1(gt_all, pred), c,
            gap_recovery(gt_all, c, hiddens), fragmentation(gt_all, c),
            classify_switches(gt_all, c, hidden=hiddens),
        ))
        print(f"[{name}] done in {time.time() - t0:.0f}s", flush=True)
    print(f"\nORACLE detections on {', '.join(seqs)} (visibility < {MIN_VISIBILITY} withheld)\n")
    print(main_table(runs)); print()
    print("Occlusion-gap recovery (recovered/reappearances):"); print(gap_table(runs)); print()
    print("ID-switch taxonomy:"); print(taxonomy_table(runs)); print()
    for r in runs:
        print(f"{r.name}: fragmentation per GT track  median={r.frag.median:.1f}  p90={r.frag.p90:.1f}")


if __name__ == "__main__":
    main(sys.argv[1:] or ["MOT17-02", "MOT17-04", "MOT17-09"])
