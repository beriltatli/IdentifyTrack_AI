from eval.clear import clear_match
from eval.taxonomy import classify_switches
from tests.helpers import build, static_track


def test_one_swap_one_gap_reassign_one_drift_are_separated():
    n = 100
    # --- swap: D (x=0) and E (x=100) exchange preds 1<->2 at frame 20
    # --- gap-reassign: C (x=300) visible 0-9 and 30-39, absent between; comes back as a new id
    # --- drift: A (x=500) leaves at 30; its pred P keeps going and lands on B (x=700),
    #            whose own pred Q vanished. B is visible throughout.
    ids = {"D": 1, "E": 2, "C": 3, "A": 4, "B": 5}
    gt = build({
        ids["D"]: static_track(0, range(40)), ids["E"]: static_track(100, range(40)),
        ids["C"]: {**static_track(300, range(10)), **static_track(300, range(30, 40))},
        ids["A"]: static_track(500, range(30)), ids["B"]: static_track(700, range(60)),
    }, n)
    pr = build({
        11: {**static_track(0, range(20)), **static_track(100, range(20, 40))},   # swap
        12: {**static_track(100, range(20)), **static_track(0, range(20, 40))},
        13: static_track(300, range(10)), 14: static_track(300, range(30, 40)),   # gap-reassign
        15: {**static_track(500, range(30)), **static_track(700, range(30, 60))}, # drift (P)
        16: static_track(700, range(30)),                                         # Q dies at 30
    }, n)
    c = clear_match(gt, pr)
    t = classify_switches(gt, c)
    assert c.idsw == 4  # swap x2 (one incident) + gap-reassign + drift
    assert t.counts == {"swap": 2, "drift": 1, "gap-reassign": 1, "other": 0}
    assert t.swap_incidents == 1
    assert sum(t.counts.values()) == c.idsw  # taxonomy and CLEAR agree by construction
    by_gt = {s.event.gt_id: s for s in t.switches}
    assert by_gt[3].cause == "gap-reassign" and by_gt[3].donor_gt is None
    assert by_gt[5].cause == "drift" and by_gt[5].donor_gt == 4


def test_fresh_id_on_visible_object_is_other():
    gt = build({1: static_track(0, range(40))}, 40)
    pr = build({1: static_track(0, range(20)), 2: static_track(0, range(20, 40))}, 40)
    t = classify_switches(gt, clear_match(gt, pr))
    assert t.counts == {"swap": 0, "drift": 0, "gap-reassign": 0, "other": 1}


def test_non_mutual_transfer_is_drift_not_swap():
    # gt1 loses its id (pred 1 -> nothing) and gt2 takes pred 1: one-directional.
    gt = build({1: static_track(0, range(40)), 2: static_track(100, range(40))}, 40)
    pr = build({1: {**static_track(0, range(20)), **static_track(100, range(20, 40))},
                2: static_track(100, range(20))}, 40)
    t = classify_switches(gt, clear_match(gt, pr))
    assert t.counts["swap"] == 0 and t.counts["drift"] == 1


def test_no_switches_no_events():
    gt = build({1: static_track(0, range(10))}, 10)
    t = classify_switches(gt, clear_match(gt, gt))
    assert t.total == 0 and t.swap_incidents == 0


def test_gt_track_that_is_hidden_in_every_frame_does_not_crash():
    # Regression (found on real MOT17): a track whose frames are ALL below the visibility
    # threshold has no 'present' frames at all, yet a detector can still switch its id.
    gt = build({1: static_track(0, range(40))}, 40)
    pr = build({1: static_track(0, range(10)), 2: static_track(0, range(25, 40))}, 40)
    hidden = {(1, f) for f in range(40)}   # 10..24 unmatched: a stretch with no visible frames
    t = classify_switches(gt, clear_match(gt, pr), hidden=hidden)
    assert t.counts["gap-reassign"] == 1 and t.total == 1
