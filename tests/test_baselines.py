import numpy as np

from baselines.greedy_iou import run_greedy_iou
from baselines.kalman_iou import run_kalman_iou
from detection.types import Detections
from eval.clear import clear_match
from eval.common import sequence_from_tracks
from eval.hota import hota
from eval.oracle import oracle_detections


def walker(frames, x0=0.0, vx=4.0, y=50.0):
    return {f: (x0 + vx * f, y, x0 + vx * f + 20, y + 40) for f in frames}


def dets_of(seq):
    return [Detections(f.boxes, np.ones(len(f.ids))) for f in seq]


def test_both_keep_one_id_on_a_straight_walker():
    gt = sequence_from_tracks({1: walker(range(40))}, 40)
    for run in (run_greedy_iou, run_kalman_iou):
        pred = run(dets_of(gt))
        assert len({int(i) for f in pred for i in f.ids}) == 1


def test_kalman_survives_a_short_occlusion_greedy_does_not():
    frames = list(range(0, 15)) + list(range(25, 45))  # 10 frames hidden, object keeps moving
    d = dets_of(sequence_from_tracks({1: walker(frames)}, 45))
    n_greedy = len({int(i) for f in run_greedy_iou(d) for i in f.ids})
    n_kalman = len({int(i) for f in run_kalman_iou(d) for i in f.ids})
    assert n_greedy == 2 and n_kalman == 1  # 4 px/frame * 10 frames = 40 px jump defeats greedy


def test_every_tracker_emits_exactly_the_input_boxes():
    """The contract that keeps DetA identical across trackers."""
    rng = np.random.default_rng(3)
    gt = sequence_from_tracks({i: walker(range(30), x0=60.0 * i) for i in range(4)}, 30)
    d = [Detections(f.boxes + rng.normal(0, 1, f.boxes.shape), np.ones(len(f.ids))) for f in gt]
    results = [hota(gt, run(d)).deta for run in (run_greedy_iou, run_kalman_iou)]
    assert abs(results[0] - results[1]) < 1e-12
    for run in (run_greedy_iou, run_kalman_iou):
        for f, det in zip(run(d), d):
            assert np.array_equal(f.boxes, det.boxes)


def test_oracle_withholds_hidden_objects():
    gt = sequence_from_tracks({1: walker(range(10)), 2: walker(range(10), y=200.0)}, 10)
    hidden = {(2, f) for f in range(3, 6)}
    n = [len(d) for d in oracle_detections(gt, hidden)]
    assert n == [2, 2, 2, 1, 1, 1, 2, 2, 2, 2]


def test_empty_frames_do_not_crash():
    empty = [Detections(np.empty((0, 4)), np.empty(0))] * 5
    assert all(len(f.ids) == 0 for f in run_greedy_iou(empty))
    assert all(len(f.ids) == 0 for f in run_kalman_iou(empty))
