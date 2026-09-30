"""Constant-velocity Kalman filter for a bounding box, written out (no filterpy).

State  x = [cx, cy, s, r, vcx, vcy, vs]
    cx, cy   box centre
    s        scale = sqrt(w*h)   (a *linear* size, so noise can be proportional to it)
    r        aspect ratio = w/h  (given no velocity: aspect changes are noise, not motion)
Measurement z = [cx, cy, s, r]

Model (dt = 1 frame):
    predict:  x' = F x                     P' = F P F^T + Q
    update :  y = z - H x'                 (innovation)
              S = H P' H^T + R             (innovation covariance)
              K = P' H^T S^-1              (Kalman gain)
              x = x' + K y                 P = (I - K H) P'
"""
from __future__ import annotations

import numpy as np

STATE_DIM, MEAS_DIM = 7, 4


def xyxy_to_z(box: np.ndarray) -> np.ndarray:
    w, h = box[2] - box[0], box[3] - box[1]
    return np.array([box[0] + w / 2, box[1] + h / 2, np.sqrt(w * h), w / h])


def state_to_xyxy(x: np.ndarray) -> np.ndarray:
    cx, cy, s, r = x[:4]
    w, h = s * np.sqrt(r), s / np.sqrt(r)
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])


class KalmanBoxFilter:
    def __init__(
        self,
        box: np.ndarray,
        sigma_pos: float = 0.05,
        sigma_vel: float = 0.01,
        sigma_meas: float = 0.05,
        sigma_aspect: float = 0.02,
    ) -> None:
        """sigma_* are fractions of the box scale s (aspect noise is absolute, r is unitless)."""
        self.sigma_pos, self.sigma_vel = sigma_pos, sigma_vel
        self.sigma_meas, self.sigma_aspect = sigma_meas, sigma_aspect
        z = xyxy_to_z(np.asarray(box, dtype=np.float64))
        self.x = np.concatenate([z, np.zeros(3)])
        s = z[2]
        # Position is known to about one measurement sigma; velocity is unknown -> large variance.
        self.P = np.diag(
            [(2 * sigma_meas * s) ** 2] * 3 + [sigma_aspect**2] + [(10 * sigma_vel * s) ** 2] * 3
        )
        self.F = np.eye(STATE_DIM)
        self.F[0, 4] = self.F[1, 5] = self.F[2, 6] = 1.0  # position += velocity
        self.H = np.zeros((MEAS_DIM, STATE_DIM))
        self.H[:4, :4] = np.eye(4)

    # ---- noise ----------------------------------------------------------------------------
    def _Q(self) -> np.ndarray:
        # Process noise: how much the *true* motion may deviate from constant velocity per frame.
        s = max(self.x[2], 1e-6)
        pos, vel = self.sigma_pos * s, self.sigma_vel * s
        return np.diag([pos**2] * 3 + [self.sigma_aspect**2 * 0.01] + [vel**2] * 3)

    def _R(self) -> np.ndarray:
        # Observation noise scales with box size: a detector that is off by ~5% of the object's
        # size is off by 3 px on a 60 px person and 30 px on a 600 px one. A fixed pixel R would
        # over-trust far-away (small) boxes and under-trust close (large) ones, which corrupts
        # both the Kalman gain and the Mahalanobis gate.
        s = max(self.x[2], 1e-6)
        m = self.sigma_meas * s
        return np.diag([m**2, m**2, m**2, self.sigma_aspect**2])

    # ---- filter steps ---------------------------------------------------------------------
    def predict(self) -> np.ndarray:
        """Advance one frame. Also what a *lost* track does every frame while coasting."""
        self.x = self.F @ self.x
        self.x[2] = max(self.x[2], 1e-3)  # scale must stay positive under long coasting
        self.P = self.F @ self.P @ self.F.T + self._Q()
        return state_to_xyxy(self.x)

    def update(self, box: np.ndarray) -> np.ndarray:
        z = xyxy_to_z(np.asarray(box, dtype=np.float64))
        S = self.H @ self.P @ self.H.T + self._R()
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z - self.H @ self.x)
        # Joseph form keeps P symmetric positive-definite despite float rounding.
        I_KH = np.eye(STATE_DIM) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ self._R() @ K.T
        return state_to_xyxy(self.x)

    # ---- gating ---------------------------------------------------------------------------
    def mahalanobis_sq(self, boxes: np.ndarray) -> np.ndarray:
        """Squared Mahalanobis distance from the predicted measurement to each xyxy box, (N,).

        Under the model this is chi-square with 4 dof, so 9.4877 is the 95% gate.
        """
        boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
        if len(boxes) == 0:
            return np.empty(0)
        S = self.H @ self.P @ self.H.T + self._R()
        d = np.array([xyxy_to_z(b) for b in boxes]) - self.H @ self.x
        return np.einsum("ni,ij,nj->n", d, np.linalg.inv(S), d)

    @property
    def box(self) -> np.ndarray:
        return state_to_xyxy(self.x)
