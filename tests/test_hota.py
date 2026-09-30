"""HOTA / IDF1 / MOTA against hand-built tracks whose answers are derived analytically."""
import math

import numpy as np
import pytest

from eval.clear import clear_match
from eval.common import sequence_from_tracks
from eval.hota import ALPHAS, hota
from eval.idf1 import idf1
from eval.report import RunMetrics, deta_warnings
from tests.helpers import box, build, static_track


def test_perfect_tracking_is_one():
    gt = build({1: static_track(0, range(50)), 2: static_track(100, range(50))}, 50)
    r = hota(gt, gt)
    assert r.hota == pytest.approx(1.0) and r.deta == pytest.approx(1.0) and r.assa == pytest.approx(1.0)
    assert idf1(gt, gt).idf1 == pytest.approx(1.0)


def test_missing_object_hurts_deta_not_assa():
    # Pred perfectly tracks object 1, never sees object 2.
    # TP=N, FN=N, FP=0 -> DetA=0.5. The one matched pair covers both lifetimes fully -> AssA=1.
    n = 40
    gt = build({1: static_track(0, range(n)), 2: static_track(100, range(n))}, n)
    pr = build({7: static_track(0, range(n))}, n)
    r = hota(gt, pr)
    assert r.deta == pytest.approx(0.5)
    assert r.assa == pytest.approx(1.0)
    assert r.hota == pytest.approx(math.sqrt(0.5))


def test_id_split_hurts_assa_not_deta():
    # One GT of 100 frames, tracked perfectly but under id 1 for 50 frames then id 2.
    # DetA=1. Two pairs, each TPA=50, A=50/(50+50+0)=0.5 -> AssA=0.5. HOTA=sqrt(0.5).
    gt = build({1: static_track(0, range(100))}, 100)
    pr = build({1: static_track(0, range(50)), 2: static_track(0, range(50, 100))}, 100)
    r = hota(gt, pr)
    assert r.deta == pytest.approx(1.0)
    assert r.assa == pytest.approx(0.5)
    assert r.hota == pytest.approx(math.sqrt(0.5))


def test_swap_gives_one_third_assa():
    # Two objects, preds exchange targets at frame 50. All four (gt,pred) pairs: TPA=50,
    # FNA=50, FPA=50 -> A=1/3 for each -> AssA=1/3, DetA=1.
    g = build({1: static_track(0, range(100)), 2: static_track(100, range(100))}, 100)
    p = build(
        {
            1: {**static_track(0, range(50)), **static_track(100, range(50, 100))},
            2: {**static_track(100, range(50)), **static_track(0, range(50, 100))},
        },
        100,
    )
    r = hota(g, p)
    assert r.deta == pytest.approx(1.0)
    assert r.assa == pytest.approx(1 / 3)


def test_iou_threshold_sweep():
    # Same box, but pred is 60% height -> IoU exactly 0.6. TP for alphas <= 0.6 (12 of 19).
    n = 20
    gt = build({1: {f: (0, 0, 10, 10) for f in range(n)}}, n)
    pr = build({1: {f: (0, 0, 10, 6) for f in range(n)}}, n)
    r = hota(gt, pr)
    assert (r.tp > 0).sum() == 12 == int((ALPHAS <= 0.6).sum())
    assert r.hota == pytest.approx(12 / 19)


def test_mota_hides_identity_shredding():
    # The reason MOTA is not the headline: new pred ID every 10 frames on a perfectly
    # detected object. 9 switches out of 100 GT boxes -> MOTA=0.91, but AssA=0.1.
    n = 100
    gt = build({1: static_track(0, range(n))}, n)
    pr = build({k + 1: static_track(0, range(10 * k, 10 * k + 10)) for k in range(10)}, n)
    c = clear_match(gt, pr)
    r = hota(gt, pr)
    assert c.idsw == 9 and c.fp == 0 and c.fn == 0
    assert c.mota == pytest.approx(0.91)
    assert r.deta == pytest.approx(1.0)
    assert r.assa == pytest.approx(0.1)      # 10 pairs, A=10/(10+90)=0.1
    assert r.hota == pytest.approx(math.sqrt(0.1))


def test_idf1_split_track():
    # Best single pairing explains 50 of 100 frames: IDTP=50, |gt|=|pred|=100 -> 2*50/200.
    gt = build({1: static_track(0, range(100))}, 100)
    pr = build({1: static_track(0, range(50)), 2: static_track(0, range(50, 100))}, 100)
    r = idf1(gt, pr)
    assert r.idtp == 50 and r.idf1 == pytest.approx(0.5)


def test_idf1_is_global_not_greedy():
    # Pred A covers frames 0-59 of GT (60), pred B covers 60-99 (40). Only one can be paired.
    gt = build({1: static_track(0, range(100))}, 100)
    pr = build({1: static_track(0, range(60)), 2: static_track(0, range(60, 100))}, 100)
    assert idf1(gt, pr).idtp == 60


def test_empty_inputs_do_not_crash():
    e = build({}, 5)
    g = build({1: static_track(0, range(5))}, 5)
    assert hota(e, e).hota == 0.0
    assert hota(g, e).deta == 0.0
    assert idf1(g, e).idf1 == 0.0


def test_deta_spread_is_flagged_not_averaged():
    n = 20
    gt = build({1: static_track(0, range(n))}, n)
    full = build({1: static_track(0, range(n))}, n)
    half = build({1: static_track(0, range(n // 2))}, n)

    def run(name, pr):
        from eval.gaps import fragmentation, gap_recovery
        from eval.taxonomy import classify_switches
        c = clear_match(gt, pr)
        return RunMetrics(name, hota(gt, pr), idf1(gt, pr), c, gap_recovery(gt, c),
                          fragmentation(gt, c), classify_switches(gt, c))

    assert deta_warnings([run("a", full), run("b", full)]) == []
    assert len(deta_warnings([run("a", full), run("b", half)])) == 1
