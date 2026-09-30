import pytest

from eval.clear import clear_match
from eval.gaps import fragmentation, gap_recovery, summarize_fragmentation
from tests.helpers import build, static_track


def _one_gt_with_gaps():
    # 5-frame visible segments separated by gaps of 3, 10, 20, 50 frames.
    segs, t = [], 0
    for gap in (None, 3, 10, 20, 50):
        if gap:
            t += gap
        segs.append(range(t, t + 5))
        t += 5
    gt = build({1: {f: (0, 0, 10, 10) for s in segs for f in s}}, t)
    return gt, segs, t


def test_bins_and_recovery_known_answer():
    gt, segs, n = _one_gt_with_gaps()
    # pred ids per segment: 1, 1 (gap3 recovered), 2 (gap10 lost), 2 (gap20 recovered), 3 (gap50 lost)
    ids = [1, 1, 2, 2, 3]
    pr = build({i: {f: (0, 0, 10, 10) for s, k in zip(segs, ids) if k == i for f in s}
                for i in set(ids)}, n)
    res = gap_recovery(gt, clear_match(gt, pr))
    got = {b.label: (b.recovered, b.n) for b in res.bins}
    assert got == {"1-5": (1, 1), "6-15": (0, 1), "16-39": (1, 1), "40+": (0, 1)}
    assert res.long_gap_recovery == 0.0


def test_bin_boundaries():
    for gap, label in [(5, "1-5"), (6, "6-15"), (15, "6-15"), (16, "16-39"), (39, "16-39"), (40, "40+")]:
        gt = build({1: {**{f: (0, 0, 10, 10) for f in range(3)},
                        **{f: (0, 0, 10, 10) for f in range(3 + gap, 6 + gap)}}}, 6 + gap)
        pr = build({1: {f: (0, 0, 10, 10) for f in range(3)},
                    2: {f: (0, 0, 10, 10) for f in range(3 + gap, 6 + gap)}}, 6 + gap)
        res = gap_recovery(gt, clear_match(gt, pr))
        assert [b.n for b in res.bins if b.label == label] == [1], (gap, label)


def test_never_reacquired_counts_as_failure():
    gt = build({1: {**{f: (0, 0, 10, 10) for f in range(5)}, **{f: (0, 0, 10, 10) for f in range(50, 55)}}}, 55)
    pr = build({1: static_track(0, range(5))}, 55)  # tracker never sees it again
    res = gap_recovery(gt, clear_match(gt, pr))
    assert res.events[0].never_reacquired and not res.events[0].recovered
    assert res.bins[-1].n == 1 and res.bins[-1].recovered == 0


def test_no_prior_match_is_excluded_not_counted():
    gt = build({1: {**{f: (0, 0, 10, 10) for f in range(5)}, **{f: (0, 0, 10, 10) for f in range(20, 25)}}}, 25)
    pr = build({1: static_track(0, range(20, 25))}, 25)  # nothing before the gap
    res = gap_recovery(gt, clear_match(gt, pr))
    assert res.excluded_no_prior_id == 1 and res.events == []


def test_long_gap_all_recovered():
    gt = build({1: {**{f: (0, 0, 10, 10) for f in range(5)}, **{f: (0, 0, 10, 10) for f in range(60, 65)}}}, 65)
    pr = build({9: {f: (0, 0, 10, 10) for f in list(range(5)) + list(range(60, 65))}}, 65)
    assert gap_recovery(gt, clear_match(gt, pr)).long_gap_recovery == 1.0


def test_fragmentation_summary_is_median_and_p90():
    med, p90 = summarize_fragmentation([1, 1, 1, 1, 1, 2, 2, 2, 4, 20])
    assert med == 1.5 and p90 == pytest.approx(4 + 0.1 * 16)  # linear interp between 4 and 20
    # the mean (3.5) would have hidden that one track is shredded into 20 pieces
    assert summarize_fragmentation([]) == (0.0, 0.0)


def test_fragmentation_from_matches():
    gt = build({1: static_track(0, range(30)), 2: static_track(100, range(30)), 3: static_track(200, range(30))}, 30)
    pr = build({
        1: static_track(0, range(10)), 2: static_track(0, range(10, 30)),   # gt1 -> 2 ids
        3: static_track(100, range(30)),                                     # gt2 -> 1 id
    }, 30)                                                                   # gt3 never matched
    fr = fragmentation(gt, clear_match(gt, pr))
    assert fr.per_track == {1: 2, 2: 1} and fr.unmatched_tracks == 1


def test_hidden_frames_turn_annotated_occlusion_into_a_gap():
    # MOT17 style: GT is annotated every frame, but frames 5-29 have low visibility.
    from eval.taxonomy import classify_switches
    gt = build({1: static_track(0, range(40))}, 40)
    pr = build({1: static_track(0, range(5)), 2: static_track(0, range(30, 40))}, 40)
    c = clear_match(gt, pr)
    assert gap_recovery(gt, c).events == []                      # no gap without visibility info
    hidden = {(1, f) for f in range(5, 30)}
    res = gap_recovery(gt, c, hidden)
    assert [(e.gap_len, e.recovered) for e in res.events] == [(25, False)]
    assert classify_switches(gt, c, hidden=hidden).counts["gap-reassign"] == 1
    assert classify_switches(gt, c).counts["other"] == 1
