import numpy as np
import pytest

from tracker.cost import fuse_cost
from tracker.gallery import Gallery
from tracker.kalman import KalmanBoxFilter
from tracker.lifecycle import State, Track


def unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


# ---------------------------------------------------------------- cost: gate BEFORE weighting
def test_perfect_appearance_cannot_buy_back_an_impossible_motion_jump():
    maha = np.array([[1.0, 50.0]])      # 2nd detection is far outside the motion gate (9.49)
    app = np.array([[0.30, 0.0]])       # ...but looks identical to the track (distance 0)
    cost = fuse_cost(maha, app, motion_gate=9.4877, app_gate=0.5, lam=0.5)
    assert np.isinf(cost[0, 1]) and np.isfinite(cost[0, 0])


def test_perfect_motion_cannot_buy_back_a_different_looking_object():
    cost = fuse_cost(np.array([[0.0]]), np.array([[0.9]]), 9.4877, 0.5, 0.5)
    assert np.isinf(cost[0, 0])


def test_weight_only_orders_feasible_pairs():
    maha = np.array([[2.0, 8.0]])
    app = np.array([[0.4, 0.1]])
    motion_heavy = fuse_cost(maha, app, 9.4877, 0.5, lam=0.95)
    app_heavy = fuse_cost(maha, app, 9.4877, 0.5, lam=0.05)
    assert motion_heavy[0].argmin() == 0 and app_heavy[0].argmin() == 1
    assert ((0 <= motion_heavy) & (motion_heavy <= 1)).all()


def test_motion_only_when_no_appearance():
    c = fuse_cost(np.array([[1.0, 99.0]]), None, 9.4877, 0.5, 0.5)
    assert np.isfinite(c[0, 0]) and np.isinf(c[0, 1])


# ---------------------------------------------------------------- gallery
def test_gallery_distance_is_min_over_history():
    g = Gallery(size=5, policy="always")
    g.seed(unit([1, 0]))
    g.update(unit([0, 1]), 1.0, 0.0)
    d = g.distance(np.stack([unit([0, 1]), unit([1, 1])]))
    assert d[0] == pytest.approx(0.0) and d[1] == pytest.approx(1 - np.sqrt(0.5))
    assert Gallery().distance(np.stack([unit([1, 0])]))[0] == 2.0  # empty gallery: maximal


def test_gated_policy_refuses_occluded_or_weak_detections_always_does_not():
    occluder = unit([0, 1])
    gated, naive = Gallery(policy="gated", update_conf=0.6, occ_iou=0.3), Gallery(policy="always")
    for g in (gated, naive):
        g.seed(unit([1, 0]))
    # overlapping another detection (IoU 0.6): the crop contains the occluder
    assert not gated.update(occluder, score=0.9, max_overlap=0.6)
    assert naive.update(occluder, score=0.9, max_overlap=0.6)
    assert not gated.update(occluder, score=0.4, max_overlap=0.0)   # weak
    assert gated.update(unit([1, 0.1]), score=0.9, max_overlap=0.0)  # clean view accepted
    assert len(gated) == 2 and len(naive) == 2
    # the poisoned gallery now thinks the occluder IS this track; the gated one does not
    assert naive.distance(np.stack([occluder]))[0] == pytest.approx(0.0)
    assert gated.distance(np.stack([occluder]))[0] > 0.8


def test_gallery_is_bounded():
    g = Gallery(size=3, policy="always")
    for k in range(10):
        g.update(unit([1, k]), 1.0, 0.0)
    assert len(g) == 3


# ---------------------------------------------------------------- lifecycle
def mk(state=State.TENTATIVE):
    return Track(1, KalmanBoxFilter(np.array([0, 0, 10, 20.0])), Gallery(), state=state)


def test_tentative_confirms_after_n_init_and_dies_on_first_miss():
    t = mk()
    t.on_match(3)
    assert t.state is State.TENTATIVE
    t.on_match(3)
    assert t.state is State.CONFIRMED
    u = mk()
    u.on_miss(100)
    assert u.state is State.DELETED


def test_confirmed_goes_lost_then_deleted_exactly_at_buffer():
    t = mk(State.CONFIRMED)
    t.on_miss(5)
    assert t.state is State.LOST
    for _ in range(4):
        t.on_miss(5)
    assert t.state is State.LOST and t.misses == 5       # 5 misses still allowed
    t.on_miss(5)
    assert t.state is State.DELETED                      # the 6th deletes


def test_lost_track_is_reacquired_and_reports_it():
    t = mk(State.CONFIRMED)
    for _ in range(7):
        t.on_miss(100)
    assert t.state is State.LOST
    assert t.on_match(3) is True and t.state is State.CONFIRMED and t.misses == 0
    assert t.on_match(3) is False


def test_iou_rescues_a_border_clipped_box_that_mahalanobis_rejects():
    # Regression (real MOT17-09, frame 25): person at the image edge, box clipped by the
    # detector. Mahalanobis 88 > gate 60, but IoU with the prediction is 0.7.
    maha, iou = np.array([[88.0]]), np.array([[0.7]])
    assert np.isinf(fuse_cost(maha, None, 60.0, 0.3, 0.5)[0, 0])             # old behaviour
    assert np.isfinite(fuse_cost(maha, None, 60.0, 0.3, 0.5, iou)[0, 0])     # fixed


def test_still_infeasible_when_both_motion_signals_reject():
    c = fuse_cost(np.array([[500.0]]), np.array([[0.0]]), 60.0, 0.3, 0.5, np.array([[0.05]]))
    assert np.isinf(c[0, 0])  # perfect appearance still cannot buy an impossible position


def test_mahalanobis_still_reaches_a_long_coasting_track_where_iou_is_zero():
    c = fuse_cost(np.array([[20.0]]), None, 60.0, 0.3, 0.5, np.array([[0.0]]))
    assert np.isfinite(c[0, 0]) and c[0, 0] == pytest.approx(20 / 60)


def test_gate_zero_means_iou_only():
    iou = np.array([[0.8, 0.1]])
    c = fuse_cost(np.array([[0.0, 0.0]]), None, 0.0, 0.3, 0.5, iou)   # maha would say both fine
    assert np.isfinite(c[0, 0]) and c[0, 0] == pytest.approx(0.2) and np.isinf(c[0, 1])
