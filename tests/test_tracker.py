"""End to end on synthetic scenes where appearance is the only thing that can work."""
import numpy as np

from baselines.appearance_only import run_appearance_only
from baselines.kalman_iou import run_kalman_iou
from detection.types import Detections
from tracker.tracker import TrackerConfig, run_tracker


def unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


FEAT_A, FEAT_B = unit([1, 0.1, 0]), unit([0.1, 1, 0])


def box(x, y=100.0):
    return np.array([x, y, x + 30, y + 80.0])


def scene(n, objects):
    """objects: list of (feat, {frame: box}) -> per-frame Detections with score 0.9."""
    out = []
    for t in range(n):
        items = [(f, b[t]) for f, b in objects if t in b]
        if items:
            out.append(Detections(np.stack([b for _, b in items]), np.full(len(items), 0.9),
                                  np.stack([f for f, _ in items])))
        else:
            out.append(Detections(np.empty((0, 4)), np.empty(0), np.empty((0, 3))))
    return out


def ids_seen(seq):
    return sorted({int(i) for f in seq for i in f.ids})


def test_reid_after_long_occlusion_at_an_unexpected_place():
    # A walks right, vanishes for 60 frames, reappears far from the extrapolated path.
    # Motion+IoU can't connect the two sightings; appearance can.
    before = {t: box(100 + 4 * t) for t in range(20)}
    after = {t: box(40 + 4 * (t - 80)) for t in range(80, 100)}
    dets = scene(100, [(FEAT_A, {**before, **after})])
    seq, trk = run_tracker(dets, TrackerConfig(lost_buffer=100))
    assert len(ids_seen(seq)) == 1
    assert len(trk.reacquisitions) == 1 and trk.reacquisitions[0][2] == 60
    assert len(ids_seen(run_kalman_iou([Detections(d.boxes, d.scores) for d in dets]))) == 2


def test_lost_buffer_is_the_hard_limit_on_recovery():
    before = {t: box(100 + 4 * t) for t in range(20)}
    after = {t: box(40) for t in range(80, 100)}
    dets = scene(100, [(FEAT_A, {**before, **after})])
    assert len(ids_seen(run_tracker(dets, TrackerConfig(lost_buffer=100))[0])) == 1
    assert len(ids_seen(run_tracker(dets, TrackerConfig(lost_buffer=30))[0])) == 2


def test_appearance_separates_two_crossing_people():
    a = {t: box(50 + 10 * t) for t in range(30)}         # left -> right
    b = {t: box(340 - 10 * t) for t in range(30)}        # right -> left, same height: they cross
    dets = scene(30, [(FEAT_A, a), (FEAT_B, b)])
    seq, _ = run_tracker(dets, TrackerConfig(lost_buffer=10))
    by_frame_id = {}
    for t, f in enumerate(seq):
        for i, bx in zip(f.ids, f.boxes):
            by_frame_id[(t, round(float(bx[0])))] = int(i)
    id_a = by_frame_id[(0, 50)]
    assert {by_frame_id[(t, 50 + 10 * t)] for t in range(30)} == {id_a}


def test_emission_contract_only_high_confidence_boxes_are_output():
    dets = scene(10, [(FEAT_A, {t: box(100 + 3 * t) for t in range(10)})])
    dets[5] = Detections(dets[5].boxes, np.array([0.15]), dets[5].feats)   # weak frame
    seq, _ = run_tracker(dets, TrackerConfig())
    assert len(seq[5].ids) == 0                          # low box is not emitted...
    assert len(ids_seen(seq)) == 1                       # ...but it kept the track alive


def test_low_confidence_box_bridges_a_missing_detection_without_changing_output_boxes():
    dets = scene(12, [(FEAT_A, {t: box(100 + 3 * t) for t in range(12)})])
    dets[6] = Detections(dets[6].boxes, np.array([0.1]), dets[6].feats)
    seq, trk = run_tracker(dets, TrackerConfig(n_init=3))
    assert len(ids_seen(seq)) == 1 and len(trk.reacquisitions) == 0   # never even went lost


def test_appearance_only_baseline_ignores_position():
    a = {t: box(50 + 5 * t) for t in range(10)}
    jump = {t: box(700 + 5 * t) for t in range(20, 30)}   # same look, teleported
    dets = scene(30, [(FEAT_A, {**a, **jump})])
    assert len(ids_seen(run_appearance_only(dets))) == 1


def test_unknown_config_key_is_rejected():
    import pytest
    with pytest.raises(KeyError):
        TrackerConfig.from_dict({"lost_bufer": 5})
