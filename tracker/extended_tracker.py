"""
extended_tracker.py

Extension module for Project T4 and T5.

This file does NOT implement:
    - EKF
    - CoordinateManager

It only connects:
    EKF + CoordinateManager + sensor measurements
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple
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
    Return vessel NED position [N, E] closest to time_s.

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
        raise ValueError("vessel_positions must have format [[time, N, E], ...]")

    idx = int(np.argmin(np.abs(arr[:, 0] - float(time_s))))
    return arr[idx, 1:3]


# -----------------------------------------------------------------------------
# Extended tracker
# -----------------------------------------------------------------------------

class ExtendedTracker:
    """
    Fusion layer for T4 and T5.

    This is still single-target.
    For T6, false alarms should be handled properly with gating and association.
    For now, false alarms are ignored using true_target_id == -1.
    """

    def __init__(
        self,
        ekf: Any,
        coordinate_manager: Any,
    ) -> None:
        self.ekf = ekf
        self.coord = coordinate_manager

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

            # IMPORTANT:
            # For T4/T5 single-target validation, skip simulator false alarms.
            # Later in T6 this must be replaced by gating + data association.
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
            NIS value if update was applied.
        """
        sensor_id = self._sensor_id(meas)

        if sensor_id not in ("radar", "camera", "ais"):
            return None

        time_s = self._measurement_time(meas)

        self.ekf.predict_to(time_s)

        if sensor_id in ("radar", "camera"):
            z, sensor_pos = self._range_bearing_measurement(meas)

        elif sensor_id == "ais":
            if vessel_pos is None:
                raise ValueError("AIS measurement needs vessel_pos.")

            z, sensor_pos = self._ais_measurement(meas, vessel_pos)

        else:
            return None

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

        Use only when radar and camera measurements are considered simultaneous.
        """
        # Do not use false alarms in this temporary single-target version.
        if self._is_false_alarm(radar_meas) or self._is_false_alarm(camera_meas):
            return None

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
            R_parts.append(self.coord.R(sensor_id))

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

        self.nis_history.append(
            {
                "time": fusion_time,
                "sensor": "joint_radar_camera",
                "nis": nis,
            }
        )

        return nis

    # -------------------------------------------------------------------------
    # Measurement conversion
    # -------------------------------------------------------------------------

    def _range_bearing_measurement(
        self,
        meas: Dict[str, Any],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert radar/camera measurement dictionary to z and sensor position.
        """
        sensor_id = self._sensor_id(meas)

        sensor_pos = self.coord.sensor_position(sensor_id)

        z = np.array(
            [
                float(meas["range_m"]),
                float(meas["bearing_rad"]),
            ],
            dtype=float,
        )

        z[1] = wrap_angle(z[1])

        return z, sensor_pos

    def _ais_measurement(
        self,
        meas: Dict[str, Any],
        vessel_pos: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert AIS absolute NED target position to range-bearing from vessel.
        """
        vessel_pos = np.asarray(vessel_pos, dtype=float)

        target_pos = np.array(
            [
                float(meas["north_m"]),
                float(meas["east_m"]),
            ],
            dtype=float,
        )

        delta = target_pos - vessel_pos

        z = np.array(
            [
                np.linalg.norm(delta),
                math.atan2(delta[1], delta[0]),
            ],
            dtype=float,
        )

        z[1] = wrap_angle(z[1])

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
        """
        Build z_pred, H, and R, then call the EKF update.
        """
        z_pred = self.coord.h_range_bearing(self.ekf.x, sensor_pos)
        H = self.coord.H_range_bearing(self.ekf.x, sensor_pos)

        if sensor_id == "ais":
            R = self.coord.R(
                sensor_id,
                x_pred=self.ekf.x,
                vessel_pos=sensor_pos,
            )
        else:
            R = self.coord.R(sensor_id)

        nis = self._call_ekf_update(
            z=z,
            z_pred=z_pred,
            H=H,
            R=R,
        )

        self.nis_history.append(
            {
                "time": float(self.ekf.t),
                "sensor": sensor_id,
                "nis": nis,
            }
        )

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

        Expected EKF interface:
            ekf.update(z, z_pred, H, R)
        """
        innovation = np.asarray(z, dtype=float) - np.asarray(z_pred, dtype=float)

        # Wrap every bearing component: 1, 3, 5, ...
        for k in range(1, len(innovation), 2):
            innovation[k] = wrap_angle(innovation[k])

        S = H @ self.ekf.P @ H.T + R

        nis = float(innovation.T @ np.linalg.inv(S) @ innovation)

        try:
            self.ekf.update(z, z_pred, H, R)
        except TypeError:
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
        return str(meas.get("sensor_id", meas.get("sensor", ""))).lower()

    @staticmethod
    def _measurement_time(meas: Dict[str, Any]) -> float:
        return float(meas.get("time", meas.get("timestamp", meas.get("t", 0.0))))

    @staticmethod
    def _is_false_alarm(meas: Dict[str, Any]) -> bool:
        """
        Detect simulator false alarms.

        Simulator convention:
            true_target_id == -1 means false alarm.

        Some versions may use:
            target_id == -1
            is_false_alarm == True
        """
        if "true_target_id" in meas:
            return int(meas["true_target_id"]) == -1

        if "target_id" in meas:
            return int(meas["target_id"]) == -1

        if "is_false_alarm" in meas:
            return bool(meas["is_false_alarm"])

        return False

    def _save_history(
        self,
        time_s: float,
        update_type: str,
    ) -> None:
        self.history.append(
            {
                "time": float(time_s),
                "update": update_type,
                "x": np.asarray(self.ekf.x).copy(),
                "P": np.asarray(self.ekf.P).copy(),
            }
        )


# -----------------------------------------------------------------------------
# Optional helper for initializing EKF from first measurement
# -----------------------------------------------------------------------------

def initial_state_from_measurement(
    meas: Dict[str, Any],
    coordinate_manager: Any,
    vessel_pos: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Build x0, P0, t0 from radar, camera, or AIS measurement.
    """
    coord = coordinate_manager

    sensor_id = str(meas.get("sensor_id", meas.get("sensor", ""))).lower()
    time_s = float(meas.get("time", meas.get("timestamp", meas.get("t", 0.0))))

    if sensor_id in ("radar", "camera"):
        sensor_pos = coord.sensor_position(sensor_id)

        r = float(meas["range_m"])
        bearing = float(meas["bearing_rad"])

        pos = sensor_pos + np.array(
            [
                r * math.cos(bearing),
                r * math.sin(bearing),
            ],
            dtype=float,
        )

    elif sensor_id == "ais":
        pos = np.array(
            [
                float(meas["north_m"]),
                float(meas["east_m"]),
            ],
            dtype=float,
        )

    else:
        raise ValueError(
            "Initialisation only supports radar, camera, or AIS measurements."
        )

    x0 = np.array(
        [
            pos[0],
            pos[1],
            0.0,
            0.0,
        ],
        dtype=float,
    )

    P0 = np.diag(
        [
            50.0**2,
            50.0**2,
            5.0**2,
            5.0**2,
        ]
    )

    return x0, P0, time_s