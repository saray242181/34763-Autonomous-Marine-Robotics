import numpy as np


def wrap_angle(a: float) -> float:
    """Wrap angle to [-pi, pi)."""
    return (a + np.pi) % (2 * np.pi) - np.pi


class EKF:
    """
    Extended Kalman Filter for single-target tracking.

    State:
        x = [p_N, p_E, v_N, v_E]^T
    """

    def __init__(self, sigma_a: float = 0.05):
        self.x = np.array([[0.0], [0.0], [0.0], [0.0]], dtype=float)

        self.P = np.diag([
            50.0**2,
            50.0**2,
            5.0**2,
            5.0**2,
        ])

        self.t = None
        self.sigma_a = float(sigma_a)

    def predict_to(self, t: float) -> None:
        """
        Predict state to time t using constant velocity model.
        """
        t = float(t)

        if self.t is None:
            self.t = t
            return

        dt = t - self.t

        if dt <= 0:
            return

        F = self._motion_matrix(dt)
        Q = self._process_noise(dt)

        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        self.t = t

    def update(
        self,
        z: np.ndarray,
        z_pred: np.ndarray,
        H: np.ndarray,
        R: np.ndarray,
    ) -> None:
        """
        Standard EKF update.

        Args:
            z: measurement vector
            z_pred: predicted measurement
            H: measurement Jacobian
            R: measurement covariance
        """
        z = np.asarray(z, dtype=float).reshape(-1, 1)
        z_pred = np.asarray(z_pred, dtype=float).reshape(-1, 1)
        H = np.asarray(H, dtype=float)
        R = np.asarray(R, dtype=float)

        innovation = z - z_pred

        # Wrap every bearing component: index 1, 3, 5, ...
        for i in range(1, innovation.shape[0], 2):
            innovation[i, 0] = wrap_angle(innovation[i, 0])

        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ innovation

        # Joseph form for numerical stability
        I = np.eye(4)
        I_KH = I - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T

    def _motion_matrix(self, dt: float) -> np.ndarray:
        """
        Constant velocity motion model.
        """
        return np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=float)

    def _process_noise(self, dt: float) -> np.ndarray:
        """
        Process noise covariance for constant-velocity model
        using white acceleration noise.
        """
        q = self.sigma_a**2

        return q * np.array([
            [dt**4 / 4, 0, dt**3 / 2, 0],
            [0, dt**4 / 4, 0, dt**3 / 2],
            [dt**3 / 2, 0, dt**2, 0],
            [0, dt**3 / 2, 0, dt**2],
        ], dtype=float)