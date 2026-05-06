import numpy as np


def wrap_angle(a: float) -> float:
    """Wrap angle to (-pi, pi]."""
    return (a + np.pi) % (2 * np.pi) - np.pi


class EKF:
    """
    Extended Kalman Filter for single-target tracking.

    State: x = [p_N, p_E, v_N, v_E]^T
    """

    def __init__(
        self,
        sigma_q_pos: float = 0.5,
        sigma_q_vel: float = 0.1,
    ):
        self.x = np.array([[0.0], [0.0], [0.0], [0.0]])
        self.P = np.diag([50.0**2, 50.0**2, 5.0**2, 5.0**2])
        self.t = None

        self.sigma_q_pos = sigma_q_pos
        self.sigma_q_vel = sigma_q_vel

    def predict_to(self, t: float) -> None:
        """
        Predict state to time t using constant velocity model.
        """
        if self.t is None:
            self.t = t
            return

        dt = t - self.t
        if dt <= 0:
            self.t = t
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
            z: measurement vector (2,)
            z_pred: predicted measurement (2,)
            H: measurement Jacobian (2, 4)
            R: measurement covariance (2, 2)
        """
        z = np.asarray(z).reshape(-1, 1)
        z_pred = np.asarray(z_pred).reshape(-1, 1)
        H = np.asarray(H)
        R = np.asarray(R)

        innovation = z - z_pred
        if innovation.ndim > 0:
            for i in range(1, innovation.shape[0], 2):
                innovation[i, 0] = wrap_angle(innovation[i, 0])

        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ innovation
        I_KH = np.eye(4) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T

    def _motion_matrix(self, dt: float) -> np.ndarray:
        """
        Constant velocity motion model.
        x_k+1 = F * x_k where F is the state transition matrix.
        """
        return np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])

    def _process_noise(self, dt: float) -> np.ndarray:
        """
        Process noise covariance for constant velocity model.
        Uses a white noise acceleration model.
        """
        qp = self.sigma_q_pos
        qv = self.sigma_q_vel

        q_pos = qp**2 * dt
        q_vel = qv**2 * dt**3 / 3
        q_cross = qv**2 * dt**2 / 2

        Q = np.array([
            [q_pos, 0, q_cross, 0],
            [0, q_pos, 0, q_cross],
            [q_cross, 0, q_vel, 0],
            [0, q_cross, 0, q_vel],
        ])

        return Q