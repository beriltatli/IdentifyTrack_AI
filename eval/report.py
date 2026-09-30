"""Tables for the evaluation suite (Phase 1: text only; figures come later)."""
from __future__ import annotations

from dataclasses import dataclass

from eval.clear import ClearResult
from eval.gaps import FragmentationResult, GapResult
from eval.hota import HotaResult
from eval.idf1 import IdResult
from eval.taxonomy import CAUSES, TaxonomyResult

DETA_TOL = 1.0  # points; the brief: DetA must vary < 1 point across tracker variants


@dataclass(frozen=True)
class RunMetrics:
    name: str
    hota: HotaResult
    idf1: IdResult
    clear: ClearResult
    gaps: GapResult
    frag: FragmentationResult
    taxonomy: TaxonomyResult


def deta_warnings(runs: list[RunMetrics], tol: float = DETA_TOL) -> list[str]:
    """Flag (never average away) a DetA spread across runs -- it means the detector moved."""
    if len(runs) < 2:
        return []
    vals = [(r.name, 100 * r.hota.deta) for r in runs]
    spread = max(v for _, v in vals) - min(v for _, v in vals)
    if spread < tol:
        return []
    detail = ", ".join(f"{n}={v:.2f}" for n, v in vals)
    return [f"DetA spread {spread:.2f} pts >= {tol} pt tolerance ({detail}): detector not frozen?"]


def main_table(runs: list[RunMetrics]) -> str:
    head = f"{'method':<16}{'HOTA':>7}{'DetA':>7}{'AssA':>7}{'IDF1':>7}{'IDSW':>6}{'LongGap':>9}{'MOTA*':>8}"
    rows = [head, "-" * len(head)]
    for r in runs:
        lg = r.gaps.long_gap_recovery
        lg_s = "n/a" if lg is None else f"{100 * lg:.1f}%"
        rows.append(
            f"{r.name:<16}{100 * r.hota.hota:>7.1f}{100 * r.hota.deta:>7.1f}{100 * r.hota.assa:>7.1f}"
            f"{100 * r.idf1.idf1:>7.1f}{r.clear.idsw:>6}{lg_s:>9}{100 * r.clear.mota:>8.1f}"
        )
    rows.append("* MOTA is a legacy column: dominated by FP/FN, i.e. by the (frozen) detector.")
    rows += [f"WARNING: {w}" for w in deta_warnings(runs)]
    return "\n".join(rows)


def gap_table(runs: list[RunMetrics]) -> str:
    labels = [b.label for b in runs[0].gaps.bins]
    head = f"{'method':<16}" + "".join(f"{'gap ' + l:>16}" for l in labels)
    rows = [head, "-" * len(head)]
    for r in runs:
        cells = []
        for b in r.gaps.bins:
            cells.append("n/a" if b.rate is None else f"{b.recovered}/{b.n} ({100 * b.rate:.0f}%)")
        rows.append(f"{r.name:<16}" + "".join(f"{c:>16}" for c in cells))
    return "\n".join(rows)


def taxonomy_table(runs: list[RunMetrics]) -> str:
    head = f"{'method':<16}" + "".join(f"{c:>14}" for c in CAUSES) + f"{'total':>8}"
    rows = [head, "-" * len(head)]
    for r in runs:
        rows.append(
            f"{r.name:<16}" + "".join(f"{r.taxonomy.counts[c]:>14}" for c in CAUSES)
            + f"{r.taxonomy.total:>8}"
        )
    rows.append("(a swap incident = 2 events)")
    return "\n".join(rows)
