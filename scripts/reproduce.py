"""One command regenerates every number in the README:  python -m scripts.reproduce

  1. unit tests            2. main comparison (real detections)   3. ablations + oracle
  4. qualitative figure    5. success-criteria check (computed, never hand-edited)
Writes results/main.json, results/sweeps.json, results/criteria.md, figures/qualitative.png.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

from eval.report import deta_warnings, gap_table, main_table, taxonomy_table
from scripts import make_figures, sweep
from scripts.evaluate import ALL, evaluate
from scripts.runlib import load_config, load_pool, metrics_row

OUT = Path("results")


def criteria(main: dict, sweeps: dict) -> str:
    k, o = main["kalman_iou"], main["ours"]
    detas = [m["DetA"] for m in main.values()]
    or_ = {r["label"]: r for r in sweeps["oracle"]}
    lines = []

    def row(ok: bool, text: str) -> None:
        lines.append(f"| {'PASS' if ok else '**MISSED**'} | {text} |")

    row(o["HOTA"] >= 55, f"HOTA >= 55: ours = {o['HOTA']:.1f} (DetA {o['DetA']:.1f}, AssA {o['AssA']:.1f})")
    lg = o["long_gap_rate"]
    row(lg is not None and lg >= 60, f"long-gap (40+) recovery >= 60%: ours = {lg:.1f}% ({o['long_gap_recovered']}/{o['long_gap_n']}) "
        f"vs Kalman+IoU {k['long_gap_rate']:.1f}% ({k['long_gap_recovered']}/{k['long_gap_n']})")
    drop = 100 * (1 - o["IDSW"] / k["IDSW"])
    row(drop >= 30, f"ID switches >= 30% lower than Kalman+IoU: {o['IDSW']} vs {k['IDSW']} ({drop:+.1f}% reduction)")
    row(max(detas) - min(detas) < 1.0, f"DetA varies < 1 pt across variants: spread = {max(detas) - min(detas):.2f}")
    row(bool(or_), "oracle run reported: AssA gap oracle - real (ours) = "
        f"{or_['ours [oracle]']['AssA'] - or_['ours [real dets]']['AssA']:+.1f} pts")
    return "| status | criterion |\n|---|---|\n" + "\n".join(lines)


def main() -> None:
    t0 = time.time()
    print("== 1/5 unit tests ==", flush=True)
    r = subprocess.run([sys.executable, "-m", "pytest", "-q"], capture_output=True, text=True)
    print(r.stdout.strip().splitlines()[-1])
    tests_ok = r.returncode == 0

    print("\n== 2/5 main comparison ==", flush=True)
    cfg = load_config()
    pool = load_pool(cfg)
    runs, fps = evaluate(cfg, ALL, pool=pool)
    print(main_table(runs), "\n\n", gap_table(runs), "\n\n", taxonomy_table(runs), sep="")
    main = {}
    for x in runs:
        main[x.name] = metrics_row(x, fps[x.name])
    OUT.mkdir(exist_ok=True)
    (OUT / "main.json").write_text(json.dumps(main, indent=1))

    print("\n== 3/5 ablations + oracle ==", flush=True)
    sweep.main(["gallery", "appearance", "lost", "lambda", "reid", "oracle"])
    sweeps = json.loads((OUT / "sweeps.json").read_text())

    print("\n== 3b/5 second frozen detector (YOLOv8s) ==", flush=True)
    if Path(cfg["detector"]["yolo_weights"]).exists():
        from scripts import yolo_experiment
        yolo_experiment.main("color_hist")
        yolo_experiment.main("resnet18")
    else:
        print(f"SKIPPED: {cfg['detector']['yolo_weights']} not found (download yolov8s.pt, see README)")

    print("\n== 4/5 qualitative figure ==", flush=True)
    make_figures.main()

    print("\n== 5/5 success criteria ==", flush=True)
    text = criteria(main, sweeps) + f"\n\nUnit tests: {'all pass' if tests_ok else 'FAILURES'}.\n"
    (OUT / "criteria.md").write_text(text)
    print(text)
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
