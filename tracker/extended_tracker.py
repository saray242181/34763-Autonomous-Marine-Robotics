"""
extended_tracker.py

Extension module for Project T4 and T5.

This file does NOT implement a new EKF.
It is only a fusion layer that can be plugged into the EKF from T3 and the
coordinate manager from T2.

Expected EKF interface from ekf.py:
    ekf.x
    ekf.P
    ekf.t
    ekf.predict_to(t)
    ekf.update(z, z_pred, H, R)

If your teammate names the functions differently, only the small calls inside
_process_range_bearing_update() need to be adapted.

State convention:
    x = [p_N, p_E, v_N, v_E]^T

Bearing convention:
    bearing = atan2(delta_E, delta_N), measured clockwise from North.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import math
import numpy as np


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def wrap_angle(angle: float | np.ndarray) -> float | np.ndarray:
    """Wrap angle(s) to [-pi, pi)."""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def block_diag(*matrices: np.ndarray) -> np.ndarray:
    """Small scipy-free block diagonal helper."""
    rows = sum(m.shape[0] for m in matrices)
    cols = sum(m.shape[1] for m in matrices)

    out = np.zeros((rows, cols), dtype=float)
    r = 0
    c = 0

    for m in matrices:
        rr, cc = m.shape
        out[r:r + rr, c:c + cc] = m
        r += rr
        c += cc

    return out


def closest_vessel_position(
    time_s: float,
    vessel_positions: Optional[np.ndarray],
) -> Optional[np.ndarray]:
    """
    Return the vessel NED position [N, E] closest to time_s.

    Expected vessel_positions format:
        [[time, N, E],
         [time, N, E],
         ...]
    """
    if vessel_positions is None:
        return None

    arr = np.asarray(vessel_positions, dtype=float)

    if arr.size == 0:
        return None

    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError(
            "vessel_positions must have format [[time, N, E], ...]"
        )

    idx = int(np.argmin(np.abs(arr[:, 0] - float(time_s))))
    return arr[idx, 1:3]


# -----------------------------------------------------------------------------
# Default coordinate helper
# -----------------------------------------------------------------------------

@dataclass
class SensorNoise:
    sigma_range: float
    sigma_bearing: float


class DefaultCoordinateManager:
    """
    Minimal coordinate manager for T4/T5.

    Later, when T2 is finished, you can replace this class with your teammate's
    coordinate manager as long as it provides equivalent methods:

        sensor_position(sensor_id, vessel_pos=None)
        h_range_bearing(x, sensor_pos)
        H_range_bearing(x, sensor_pos)
        R_for_sensor(sensor_id, x_pred=None, vessel_pos=None)
    """

    def __init__(
        self,
        radar_pos: Iterable[float] = (0.0, 0.0),
        camera_pos: Iterable[float] = (-80.0, 120.0),
        radar_noise: SensorNoise = SensorNoise(5.0, np.deg2rad(0.3)),
        camera_noise: SensorNoise = SensorNoise(8.0, np.deg2rad(0.15)),
        sigma_ais_pos: float = 4.0,
        sigma_gnss_pos: float = 2.0,
    ) -> None:
        self.radar_pos = np.asarray(radar_pos, dtype=float)
        self.camera_pos = np.asarray(camera_pos, dtype=float)

        self.R_radar = np.diag([
            radar_noise.sigma_range**2,
            radar_noise.sigma_bearing**2,
        ])

        self.R_camera = np.diag([
            camera_noise.sigma_range**2,
            camera_noise.sigma_bearing**2,
        ])

        self.sigma_ais_pos = float(sigma_ais_pos)
        self.sigma_gnss_pos = float(sigma_gnss_pos)

    def sensor_position(
        self,
        sensor_id: str,
        vessel_pos: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        sensor_id = sensor_id.lower()

        if sensor_id == "radar":
            return self.radar_pos

        if sensor_id == "camera":
            return self.camera_pos

        if sensor_id == "ais":
            if vessel_pos is None:
                raise ValueError("AIS update needs vessel_pos from GNSS.")
            return np.asarray(vessel_pos, dtype=float)

        raise ValueError(f"Unsupported sensor_id: {sensor_id}")

    @staticmethod
    def h_range_bearing(x: np.ndarray, sensor_pos: np.ndarray) -> np.ndarray:
        """
        Measurement model:

            z = [range, bearing]

        where bearing = atan2(delta_E, delta_N).
        """
        dN = float(x[0] - sensor_pos[0])
        dE = float(x[1] - sensor_pos[1])

        r = math.hypot(dN, dE)
        bearing = math.atan2(dE, dN)

        return np.array([r, bearing], dtype=float)

    @staticmethod
    def H_range_bearing(x: np.ndarray, sensor_pos: np.ndarray) -> np.ndarray:
        """
        Jacobian of range-bearing model with respect to:

            x = [p_N, p_E, v_N, v_E]
        """
        dN = float(x[0] - sensor_pos[0])
        dE = float(x[1] - sensor_pos[1])

        q = dN**2 + dE**2
        q = max(q, 1e-9)

        r = math.sqrt(q)

        H = np.zeros((2, 4), dtype=float)
        H[0, 0] = dN / r
        H[0, 1] = dE / r
        H[1, 0] = -dE / q
        H[1, 1] = dN / q

        return H

    def R_for_sensor(
        self,
        sensor_id: str,
        x_pred: Optional[np.ndarray] = None,
        vessel_pos: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Return measurement covariance R.

        Radar and camera are already range-bearing sensors.

        AIS is originally an absolute NED position measurement. For this project,
        it is converted into implied range-bearing relative to the vessel, so its
        covariance is approximated by first-order propagation.
        """
        sensor_id = sensor_id.lower()

        if sensor_id == "radar":
            return self.R_radar

        if sensor_id == "camera":
            return self.R_camera

        if sensor_id == "ais":
            return self._ais_range_bearing_covariance(x_pred, vessel_pos)

        raise ValueError(f"Unsupported sensor_id: {sensor_id}")

    def _ais_range_bearing_covariance(
        self,
        x_pred: Optional[np.ndarray],
        vessel_pos: Optional[np.ndarray],
    ) -> np.ndarray:
        total_position_variance = self.sigma_ais_pos**2 + self.sigma_gnss_pos**2

        if x_pred is None or vessel_pos is None:
            return np.diag([
                total_position_variance,
                np.deg2rad(0.3)**2,
            ])

        vessel_pos = np.asarray(vessel_pos, dtype=float)

        dN = float(x_pred[0] - vessel_pos[0])
        dE = float(x_pred[1] - vessel_pos[1])

        q = dN**2 + dE**2
        q = max(q, 1e-9)

        r = math.sqrt(q)

        J = np.array([
            [dN / r, dE / r],
            [-dE / q, dN / q],
        ])

        R_position = total_position_variance * np.eye(2)

        return J @ R_position @ J.T


# -----------------------------------------------------------------------------
# Extended tracker
# -----------------------------------------------------------------------------

class ExtendedTracker:
    """
    Fusion layer for T4 and T5.

    It assumes one EKF track already exists. For T6/T7, this same class can be
    used inside each track object after data association chooses the measurement
    assigned to that track.
    """

    def __init__(
        self,
        ekf: Any,
        coordinate_manager: Optional[Any] = None,
    ) -> None:
        self.ekf = ekf
        self.coord = coordinate_manager or DefaultCoordinateManager()
        self.history: List[Dict[str, Any]] = []
        self.nis_history: List[Dict[str, Any]] = []

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def process_measurements_sequential(
        self,
        measurements: Sequence[Dict[str, Any]],
        vessel_positions: Optional[np.ndarray] = None,
        allowed_sensors: Tuple[str, ...] = ("radar", "camera", "ais"),
        ignore_false_alarms: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Main loop for T4 and T5.

        It processes measurements in time order:
            predict to measurement time
            update using that sensor

        For Scenario B:
            allowed_sensors=("radar", "camera")

        For Scenario C:
            allowed_sensors=("radar", "camera", "ais")
        """
        allowed = {s.lower() for s in allowed_sensors}
        ordered_measurements = sorted(
            measurements,
            key=lambda m: self._measurement_time(m),
        )

        for meas in ordered_measurements:
            sensor_id = self._sensor_id(meas)

            if sensor_id not in allowed:
                continue

            if sensor_id == "gnss":
                continue

            if ignore_false_alarms and self._is_false_alarm(meas):
                continue

            vessel_pos = None
            if sensor_id == "ais":
                vessel_pos = closest_vessel_position(
                    self._measurement_time(meas),
                    vessel_positions,
                )

                if vessel_pos is None:
                    continue

            self.update_one(meas, vessel_pos=vessel_pos)

        return self.history

    def update_one(
        self,
        meas: Dict[str, Any],
        vessel_pos: Optional[np.ndarray] = None,
    ) -> Optional[float]:
        """
        Process one radar, camera, or AIS measurement.

        Returns:
            NIS value if update was applied, otherwise None.
        """
        sensor_id = self._sensor_id(meas)

        if sensor_id not in ("radar", "camera", "ais"):
            return None

        time_s = self._measurement_time(meas)
        self.ekf.predict_to(time_s)

        if sensor_id in ("radar", "camera"):
            z, sensor_pos = self._range_bearing_measurement(meas)

        else:
            if vessel_pos is None:
                raise ValueError("AIS measurement needs vessel_pos.")
            z, sensor_pos = self._ais_measurement(meas, vessel_pos)

        nis = self._process_range_bearing_update(
            z=z,
            sensor_pos=sensor_pos,
            sensor_id=sensor_id,
        )

        self._save_history(time_s, sensor_id)
        return nis

    def update_joint_radar_camera(
        self,
        radar_meas: Dict[str, Any],
        camera_meas: Dict[str, Any],
    ) -> Optional[float]:
        """
        Optional T4 joint update.

        Use this only when radar and camera measurements are treated as
        simultaneous, or after you have predicted the EKF to a common fusion time.
        """
        radar_time = self._measurement_time(radar_meas)
        camera_time = self._measurement_time(camera_meas)

        fusion_time = max(radar_time, camera_time)
        self.ekf.predict_to(fusion_time)

        z_parts = []
        h_parts = []
        H_parts = []
        R_parts = []

        for meas in (radar_meas, camera_meas):
            sensor_id = self._sensor_id(meas)

            if sensor_id not in ("radar", "camera"):
                raise ValueError("Joint update only supports radar and camera.")

            z, sensor_pos = self._range_bearing_measurement(meas)

            z_parts.append(z)
            h_parts.append(self.coord.h_range_bearing(self.ekf.x, sensor_pos))
            H_parts.append(self.coord.H_range_bearing(self.ekf.x, sensor_pos))
            R_parts.append(self.coord.R_for_sensor(sensor_id))

        z_joint = np.hstack(z_parts)
        h_joint = np.hstack(h_parts)
        H_joint = np.vstack(H_parts)
        R_joint = block_diag(*R_parts)

        nis = self._call_ekf_update(
            z=z_joint,
            z_pred=h_joint,
            H=H_joint,
            R=R_joint,
        )

        self._save_history(fusion_time, "joint_radar_camera")
        self.nis_history.append({
            "time": fusion_time,
            "sensor": "joint_radar_camera",
            "nis": nis,
        })

        return nis

    # -------------------------------------------------------------------------
    # Measurement conversion
    # -------------------------------------------------------------------------

    def _range_bearing_measurement(
        self,
        meas: Dict[str, Any],
    ) -> Tuple[np.ndarray, np.ndarray]:
        sensor_id = self._sensor_id(meas)

        sensor_pos = self.coord.sensor_position(sensor_id)

        z = np.array([
            float(meas["range_m"]),
            float(meas["bearing_rad"]),
        ])

        return z, sensor_pos

    def _ais_measurement(
        self,
        meas: Dict[str, Any],
        vessel_pos: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert AIS absolute NED target position to range-bearing relative to
        the vessel position.
        """
        vessel_pos = np.asarray(vessel_pos, dtype=float)

        target_pos = np.array([
            float(meas["north_m"]),
            float(meas["east_m"]),
        ])

        delta = target_pos - vessel_pos

        z = np.array([
            np.linalg.norm(delta),
            math.atan2(delta[1], delta[0]),
        ])

        sensor_pos = self.coord.sensor_position(
            "ais",
            vessel_pos=vessel_pos,
        )

        return z, sensor_pos

    # -------------------------------------------------------------------------
    # EKF update wrapper
    # -------------------------------------------------------------------------

    def _process_range_bearing_update(
        self,
        z: np.ndarray,
        sensor_pos: np.ndarray,
        sensor_id: str,
    ) -> float:
        z_pred = self.coord.h_range_bearing(self.ekf.x, sensor_pos)
        H = self.coord.H_range_bearing(self.ekf.x, sensor_pos)

        if sensor_id == "ais":
            R = self.coord.R_for_sensor(
                sensor_id,
                x_pred=self.ekf.x,
                vessel_pos=sensor_pos,
            )
        else:
            R = self.coord.R_for_sensor(sensor_id)

        nis = self._call_ekf_update(
            z=z,
            z_pred=z_pred,
            H=H,
            R=R,
        )

        self.nis_history.append({
            "time": float(self.ekf.t),
            "sensor": sensor_id,
            "nis": nis,
        })

        return nis

    def _call_ekf_update(
        self,
        z: np.ndarray,
        z_pred: np.ndarray,
        H: np.ndarray,
        R: np.ndarray,
    ) -> float:
        """
        Calls the EKF update.

        Preferred EKF interface:
            ekf.update(z, z_pred, H, R)

        If your teammate's EKF has another interface, change only this function.
        """
        innovation = np.asarray(z, dtype=float) - np.asarray(z_pred, dtype=float)

        for k in range(1, len(innovation), 2):
            innovation[k] = wrap_angle(innovation[k])

        S = H @ self.ekf.P @ H.T + R
        nis = float(innovation.T @ np.linalg.inv(S) @ innovation)

        try:
            self.ekf.update(z, z_pred, H, R)
        except TypeError:
            # Alternative common interface:
            # ekf.update(z, h_function, H_function, R)
            raise TypeError(
                "Your EKF update interface is different. "
                "Adapt ExtendedTracker._call_ekf_update(). "
                "Expected: ekf.update(z, z_pred, H, R)."
            )

        return nis

    # -------------------------------------------------------------------------
    # Small helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _sensor_id(meas: Dict[str, Any]) -> str:
        return str(
            meas.get("sensor_id", meas.get("sensor", ""))
        ).lower()

    @staticmethod
    def _measurement_time(meas: Dict[str, Any]) -> float:
        return float(
            meas.get("time", meas.get("timestamp", meas.get("t", 0.0)))
        )

    @staticmethod
    def _is_false_alarm(meas: Dict[str, Any]) -> bool:
        if "is_false_alarm" in meas:
            return bool(meas["is_false_alarm"])

        if "true_target_id" in meas:
            return int(meas["true_target_id"]) == -1

        if "target_id" in meas:
            return int(meas["target_id"]) == -1

        return False

    def _save_history(self, time_s: float, update_type: str) -> None:
        self.history.append({
            "time": float(time_s),
            "update": update_type,
            "x": np.asarray(self.ekf.x).copy(),
            "P": np.asarray(self.ekf.P).copy(),
        })


# -----------------------------------------------------------------------------
# Optional helper for initializing an EKF from the first measurement
# -----------------------------------------------------------------------------

def initial_state_from_measurement(
    meas: Dict[str, Any],
    coordinate_manager: Optional[Any] = None,
    vessel_pos: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Build x0, P0, t0 from a radar, camera, or AIS measurement.

    This is optional. Use it only if your teammate's tracker.py needs a simple
    way to create the first EKF state.
    """
    coord = coordinate_manager or DefaultCoordinateManager()

    sensor_id = str(meas.get("sensor_id", meas.get("sensor", ""))).lower()
    time_s = float(meas.get("time", meas.get("timestamp", meas.get("t", 0.0))))

    if sensor_id in ("radar", "camera"):
        sensor_pos = coord.sensor_position(sensor_id)

        r = float(meas["range_m"])
        bearing = float(meas["bearing_rad"])

        pos = sensor_pos + np.array([
            r * math.cos(bearing),
            r * math.sin(bearing),
        ])

    elif sensor_id == "ais":
        pos = np.array([
            float(meas["north_m"]),
            float(meas["east_m"]),
        ])

    else:
        raise ValueError(
            "Initialisation only supports radar, camera, or AIS measurements."
        )

    x0 = np.array([
        pos[0],
        pos[1],
        0.0,
        0.0,
    ])

    P0 = np.diag([
        50.0**2,
        50.0**2,
        5.0**2,
        5.0**2,
    ])

    return x0, P0, time_s
