"""Draw the three README charts straight from results/main.json.

Why a script and not hand-made images: the numbers come from the experiment, so the charts can
never drift away from the results table. Re-run after `scripts.reproduce`:

    .venv/bin/python -m scripts.make_readme_figures

Writes figures/hota_breakdown.png, figures/gap_recovery.png, figures/id_switch_taxonomy.png.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to files only, no window needed
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures"

# One colour per method, reused in every chart so the eye can follow a method across figures.
METHODS = [
    ("greedy_iou", "Greedy IoU", "#9aa5b1"),
    ("kalman_iou", "Kalman + IoU", "#2f7ed8"),
    ("appearance_only", "Appearance only", "#e0a030"),
    ("ours", "IdentifyTrack (full)", "#d6336c"),
]


def _style(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)


def hota_breakdown(res: dict) -> None:
    """HOTA split into DetA and AssA: DetA must be flat (same boxes), AssA is where trackers differ."""
    fig, ax = plt.subplots(figsize=(8, 4), dpi=160)
    x = np.arange(len(METHODS))
    w = 0.26
    for k, (key, label, color) in enumerate([("HOTA", "HOTA", "#222831"), ("DetA", "DetA", "#b8c0cc"),
                                              ("AssA", "AssA", "#d6336c")]):
        vals = [res[m][key] for m, _, _ in METHODS]
        bars = ax.bar(x + (k - 1) * w, vals, w, label=label, color=color)
        ax.bar_label(bars, fmt="%.1f", fontsize=7, padding=2)
    ax.set_xticks(x, [lbl for _, lbl, _ in METHODS], fontsize=8)
    ax.set_ylim(0, 70)
    ax.set_ylabel("score")
    ax.set_title("HOTA = √(DetA × AssA): detection is identical, association differs", fontsize=10)
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper left")
    _style(ax)
    fig.tight_layout()
    fig.savefig(OUT / "hota_breakdown.png", facecolor="white")
    plt.close(fig)


def gap_recovery(res: dict) -> None:
    """Share of reappearing objects that got their ORIGINAL id back, by how long they were hidden."""
    bins = ["1-5", "6-15", "16-39", "40+"]
    fig, ax = plt.subplots(figsize=(8, 4), dpi=160)
    for key, label, color in METHODS:
        rates = [100 * res[key]["gap_bins"][b][0] / res[key]["gap_bins"][b][1] for b in bins]
        ax.plot(bins, rates, marker="o", lw=2.4 if key == "ours" else 1.8, label=label, color=color)
        ax.annotate(f"{rates[-1]:.0f}%", (3, rates[-1]), textcoords="offset points", xytext=(8, -3),
                    fontsize=8, color=color)
    ax.set_xlim(-0.2, 3.5)
    ax.set_ylim(0, 60)
    ax.set_xlabel("frames the object was hidden (occlusion gap)")
    ax.set_ylabel("original ID recovered (%)")
    ax.set_title("Every tracker collapses as the gap grows", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    _style(ax)
    fig.tight_layout()
    fig.savefig(OUT / "gap_recovery.png", facecolor="white")
    plt.close(fig)


def id_switch_taxonomy(res: dict) -> None:
    """Which KIND of identity mistake each tracker makes (stacked, same classification order as eval)."""
    kinds = [("swap", "#d6336c"), ("drift", "#2f7ed8"), ("gap-reassign", "#e0a030"), ("other", "#b8c0cc")]
    fig, ax = plt.subplots(figsize=(8, 4), dpi=160)
    y = np.arange(len(METHODS))
    left = np.zeros(len(METHODS))
    for kind, color in kinds:
        vals = np.array([res[m]["taxonomy"][kind] for m, _, _ in METHODS], dtype=float)
        ax.barh(y, vals, left=left, label=kind, color=color)
        left += vals
    for yi, total in zip(y, left):
        ax.text(total + 15, yi, f"{int(total)}", va="center", fontsize=8)
    ax.set_yticks(y, [lbl for _, lbl, _ in METHODS], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("ID-switch events")
    ax.set_title("What kind of mistake? Appearance trades memory errors for drift", fontsize=10)
    ax.legend(frameon=False, fontsize=8, ncol=4, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUT / "id_switch_taxonomy.png", facecolor="white")
    plt.close(fig)


def main() -> None:
    res = json.loads((ROOT / "results" / "main.json").read_text())
    OUT.mkdir(exist_ok=True)
    hota_breakdown(res)
    gap_recovery(res)
    id_switch_taxonomy(res)
    print("wrote", ", ".join(p.name for p in sorted(OUT.glob("*.png"))))


if __name__ == "__main__":
    main()
