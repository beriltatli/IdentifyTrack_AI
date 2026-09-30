"""Kalman filter against motion whose true state is known by construction."""
import numpy as np
import pytest

from tracker.kalman import KalmanBoxFilter, state_to_xyxy, xyxy_to_z


def truth_box(t: int, cx0=100.0, cy0=200.0, vx=3.0, vy=-2.0, s0=50.0, vs=0.5, r=0.5):
    """Constant velocity on centre and scale, constant aspect."""
    cx, cy, s = cx0 + vx * t, cy0 + vy * t, s0 + vs * t
    w, h = s * np.sqrt(r), s / np.sqrt(r)
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])


def test_box_state_roundtrip():
    b = np.array([10.0, 20.0, 50.0, 100.0])
    assert np.allclose(state_to_xyxy(np.concatenate([xyxy_to_z(b), np.zeros(3)])), b)


def test_noiseless_constant_velocity_is_recovered():
    kf = KalmanBoxFilter(truth_box(0))
    for t in range(1, 41):
        kf.predict()
        kf.update(truth_box(t))
    # velocities learned exactly, and state sits on the truth
    assert kf.x[4:] == pytest.approx([3.0, -2.0, 0.5], abs=1e-3)
    assert np.allclose(kf.box, truth_box(40), atol=1e-2)
    # one-step-ahead prediction lands on the next true box
    assert np.allclose(kf.predict(), truth_box(41), atol=1e-2)


def test_coasting_extrapolates_at_learned_velocity():
    kf = KalmanBoxFilter(truth_box(0))
    for t in range(1, 31):
        kf.predict()
        kf.update(truth_box(t))
    for t in range(31, 71):  # 40 frames with no measurement (an occlusion)
        pred = kf.predict()
    assert np.allclose(pred, truth_box(70), atol=0.5)


def test_uncertainty_grows_while_coasting_and_shrinks_on_update():
    kf = KalmanBoxFilter(truth_box(0))
    for t in range(1, 11):
        kf.predict()
        kf.update(truth_box(t))
    p0 = np.trace(kf.P[:2, :2])
    for _ in range(20):
        kf.predict()
    p_coast = np.trace(kf.P[:2, :2])
    kf.update(truth_box(31))
    assert p_coast > 5 * p0 and np.trace(kf.P[:2, :2]) < p_coast


def test_filter_beats_raw_noisy_measurements():
    rng = np.random.default_rng(0)
    kf = KalmanBoxFilter(truth_box(0))
    raw_err, kf_err = [], []
    for t in range(1, 101):
        truth = truth_box(t)
        meas = truth + rng.normal(0, 2.0, 4)
        kf.predict()
        est = kf.update(meas)
        if t > 20:  # after burn-in
            raw_err.append(np.abs(meas - truth).mean())
            kf_err.append(np.abs(est - truth).mean())
    assert np.mean(kf_err) < 0.8 * np.mean(raw_err)


def test_covariance_stays_symmetric_positive_definite():
    rng = np.random.default_rng(1)
    kf = KalmanBoxFilter(truth_box(0))
    for t in range(1, 200):
        kf.predict()
        if t % 7:  # skip some updates
            kf.update(truth_box(t) + rng.normal(0, 1.0, 4))
    assert np.allclose(kf.P, kf.P.T, atol=1e-9)
    assert np.linalg.eigvalsh(kf.P).min() > 0


def test_mahalanobis_separates_true_from_far_box():
    kf = KalmanBoxFilter(truth_box(0))
    for t in range(1, 21):
        kf.predict()
        kf.update(truth_box(t))
    kf.predict()
    d = kf.mahalanobis_sq(np.stack([truth_box(21), truth_box(21) + [80, 80, 80, 80]]))
    assert d[0] < 9.4877 < d[1]  # inside / outside the chi-square(4) 95% gate
    assert kf.mahalanobis_sq(np.empty((0, 4))).shape == (0,)


def test_observation_noise_scales_with_box_size():
    """The same 6 px miss is suspicious for a small box and ordinary for a large one."""
    def gate_distance(scale: float) -> float:
        base = lambda t: truth_box(t, s0=scale, vs=0.0, vx=0.0, vy=0.0, cx0=500, cy0=500)
        kf = KalmanBoxFilter(base(0))
        for t in range(1, 15):
            kf.predict()
            kf.update(base(t))
        kf.predict()
        off = base(15) + np.array([6.0, 0, 6.0, 0])  # shift by 6 px
        return float(kf.mahalanobis_sq(off[None])[0])

    assert gate_distance(30.0) > 4 * gate_distance(300.0)
