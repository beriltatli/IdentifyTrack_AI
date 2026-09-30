"""Phase 5: ablations and sweeps. Every table is also written to results/sweeps.json.

    python -m scripts.sweep                 # everything
    python -m scripts.sweep gallery lost    # some of: gallery lost lambda appearance reid oracle
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from eval.report import DETA_TOL, deta_warnings
from scripts.evaluate import evaluate
from scripts.runlib import load_config, load_pool, metrics_row

OUT = Path("results")


def table(title: str, rows: list[dict], key: str = "name", groups: list[str] | None = None) -> str:
    h = f"{key:<26}{'HOTA':>6}{'DetA':>6}{'AssA':>6}{'IDF1':>6}{'IDSW':>6}{'40+ gap':>14}{'swap':>6}{'drift':>6}{'gap-re':>7}{'other':>6}"
    out = [f"### {title}", "", h, "-" * len(h)]
    for r in rows:
        t = r["taxonomy"]
        lg = "n/a" if r["long_gap_rate"] is None else f"{r['long_gap_recovered']}/{r['long_gap_n']} ({r['long_gap_rate']:.0f}%)"
        out.append(f"{r['label']:<26}{r['HOTA']:>6.1f}{r['DetA']:>6.1f}{r['AssA']:>6.1f}{r['IDF1']:>6.1f}{r['IDSW']:>6}{lg:>14}"
                   f"{t['swap']:>6}{t['drift']:>6}{t['gap-reassign']:>7}{t['other']:>6}")
    # DetA must be constant among runs that share one detection source. The oracle table mixes
    # two sources (GT boxes vs the real detector) on purpose, so it is checked per source.
    for g in (groups or [""]):
        sel = [r for r in rows if g in r["label"]]
        spread = max(r["DetA"] for r in sel) - min(r["DetA"] for r in sel)
        tag = f" [{g.strip('[] ')}]" if g else ""
        out.append(f"DetA spread{tag}: {spread:.2f} pts" + ("  <-- WARNING: detector not frozen?" if spread >= DETA_TOL else "  (ok, < 1 pt)"))
    return "\n".join(out)


def one(cfg, pool, label: str, ov: dict, method: str = "ours", oracle: bool = False) -> dict:
    runs, fps = evaluate(cfg, [method], oracle=oracle, overrides=ov, pool=pool)
    row = metrics_row(runs[0], fps[method])
    row["label"] = label
    return row


def main(which: list[str]) -> None:
    cfg = load_config()
    pool = load_pool(cfg)
    results: dict[str, list[dict]] = {}
    t0 = time.time()

    def section(name: str, title: str, rows: list[dict], groups: list[str] | None = None) -> None:
        results[name] = rows
        print(table(title, rows, groups=groups), "\n", flush=True)

    if "gallery" in which:
        section("gallery", "(a) Gallery update policy", [
            one(cfg, pool, "always-update", {"gallery_policy": "always"}),
            one(cfg, pool, "occlusion-gated", {"gallery_policy": "gated"}),
        ])
    if "appearance" in which:
        section("appearance", "(e) What does appearance add at all?", [
            one(cfg, pool, "motion only (no appearance)", {"use_appearance": False}),
            one(cfg, pool, "motion + appearance", {}),
        ])
    if "lost" in which:
        section("lost", "(b) Lost-buffer length (frames)", [
            one(cfg, pool, f"lost_buffer={b}", {"lost_buffer": b}) for b in (0, 10, 30, 60, 100, 200)
        ])
    if "lambda" in which:
        section("lambda", "(c) Motion weight lambda (1 = motion only, 0 = appearance only)", [
            one(cfg, pool, f"lambda_motion={l}", {"lambda_motion": l}) for l in (0.0, 0.25, 0.5, 0.75, 1.0)
        ])
    if "reid" in which:
        from scripts.calibrate import calibrate
        rows = [one(cfg, pool, f"{m} [color_hist]", {}, m) for m in ("appearance_only", "ours")]
        cfg2 = {**cfg, "reid": dict(cfg["reid"], backend="resnet18"),
                "tracker": dict(cfg["tracker"]), "baselines": {k: dict(v) for k, v in cfg["baselines"].items()}}
        cal = calibrate(cfg2)
        gate = round(cal["app_gate"], 2)
        cfg2["tracker"]["app_gate"] = gate
        cfg2["baselines"]["appearance_only"]["app_gate"] = gate
        rows += [one(cfg2, pool, f"{m} [resnet18]", {}, m) for m in ("appearance_only", "ours")]
        section("reid", f"(f) Color histogram vs ImageNet ResNet18 features (resnet app_gate={gate})", rows,
                ["[color_hist]", "[resnet18]"])
        results["reid_calibration"] = [cal]
    if "oracle" in which:
        rows = []
        for m in ("greedy_iou", "kalman_iou", "appearance_only", "ours"):
            rows.append(one(cfg, pool, f"{m} [oracle]", {}, m, oracle=True))
            rows.append(one(cfg, pool, f"{m} [real dets]", {}, m, oracle=False))
        section("oracle", "(d) Oracle (GT) detections vs real detector output", rows, ["[oracle]", "[real dets]"])
        gaps = []
        for m in ("greedy_iou", "kalman_iou", "appearance_only", "ours"):
            o = next(r for r in rows if r["label"] == f"{m} [oracle]")
            r_ = next(r for r in rows if r["label"] == f"{m} [real dets]")
            gaps.append({"method": m, "AssA_oracle": o["AssA"], "AssA_real": r_["AssA"], "gap": o["AssA"] - r_["AssA"]})
        results["oracle_assa_gap"] = gaps
        print("AssA gap, oracle - real:", *[f"{g['method']}={g['gap']:+.1f}" for g in gaps], "\n")
    OUT.mkdir(exist_ok=True)
    path = OUT / "sweeps.json"
    old = json.loads(path.read_text()) if path.exists() else {}
    path.write_text(json.dumps({**old, **results}, indent=1))
    print(f"wrote {path}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main(sys.argv[1:] or ["gallery", "appearance", "lost", "lambda", "oracle"])
