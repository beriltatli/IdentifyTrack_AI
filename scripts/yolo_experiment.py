"""Extra experiment: swap the detector for a frozen YOLOv8s (floor 0.10) and re-run everything.

With MOT17's public detections the low-confidence band does not exist (scores start at 0.4), so the
two-stage matching's second stage is inert. YOLO with a 0.10 floor produces a real band, so here
the stage can finally be tested: the last table switches it off (stage2_iou=2.0 accepts nothing).

    python -m scripts.yolo_experiment [color_hist|resnet18]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from eval.report import gap_table, main_table, taxonomy_table
from scripts.evaluate import ALL, evaluate
from scripts.runlib import load_config, load_pool, metrics_row, prepare


def main(reid: str = "color_hist") -> None:
    cfg = load_config()
    cfg["detector"] = dict(cfg["detector"], backend="yolo")
    cfg["reid"] = dict(cfg["reid"], backend=reid)
    if reid != "color_hist":
        from scripts.calibrate import calibrate
        gate = round(calibrate(cfg)["app_gate"], 2)
        cfg["tracker"]["app_gate"] = cfg["baselines"]["appearance_only"]["app_gate"] = gate
        print(f"app_gate re-calibrated for {reid} on YOLO detections: {gate}")
    t0, n = time.time(), 0
    allscores = []
    for s in cfg["data"]["sequences"]:
        dets, sec = prepare(cfg, s)
        n += len(dets)
        allscores += [d.scores for d in dets]
        print(f"{s}: {len(dets)} frames, {sum(len(d) for d in dets)} detections"
              + (f", detector+features {sec:.0f}s" if sec else " (cached)"), flush=True)
    sc = np.concatenate(allscores)
    oc = cfg["output_conf"]
    print(f"\nYOLO score band: {np.mean(sc < oc) * 100:.0f}% of detections are LOW (0.10 <= s < {oc}), "
          f"{np.mean(sc >= oc) * 100:.0f}% HIGH; median score {np.median(sc):.2f}")
    pool = load_pool(cfg)
    runs, fps = evaluate(cfg, ALL, pool=pool)
    print("\n" + main_table(runs) + "\n\n" + gap_table(runs) + "\n\n" + taxonomy_table(runs))
    out = {r.name: metrics_row(r, fps[r.name]) for r in runs}
    off, _ = evaluate(cfg, ["ours"], overrides={"stage2_iou": 2.0}, pool=pool)
    out["ours_stage2_off"] = metrics_row(off[0], None)
    a, b = out["ours"], out["ours_stage2_off"]
    print(f"\nStage 2 (low-confidence matching) ON : HOTA {a['HOTA']:.1f} AssA {a['AssA']:.1f} IDSW {a['IDSW']} 40+ {a['long_gap_recovered']}/{a['long_gap_n']}")
    print(f"Stage 2 OFF                          : HOTA {b['HOTA']:.1f} AssA {b['AssA']:.1f} IDSW {b['IDSW']} 40+ {b['long_gap_recovered']}/{b['long_gap_n']}")
    Path("results").mkdir(exist_ok=True)
    Path(f"results/yolo_{reid}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "color_hist")
