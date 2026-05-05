"""
extended_tracker.py

Helper module for Project parts T4 and T5:
- T4: radar + stereo camera EKF fusion (sequential and optional joint update)
- T5: AIS asynchronous fusion using closest GNSS/vessel position

State convention:
    x = [p_N, p_E, v_N, v_E]^T
Bearing convention:
    bearing = atan2(delta_E, delta_N), measured clockwise from North.

This file is intentionally standalone so it can be imported later from tracker.py:
    from extended_tracker import ExtendedMultiSensorTracker

How to use it in the tracker :

from extended_tracker import ExtendedMultiSensorTracker

tracker = ExtendedMultiSensorTracker()

tracker.process_measurements_sequential(
    measurements,
    vessel_positions=vessel_positions,
    allowed_sensors=("radar", "camera", "ais")
)

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple, Any
import json
import math
import numpy as np


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------

def wrap_angle(a: float | np.ndarray) -> float | np.ndarray:
    """Wrap angle(s) to [-pi, pi)."""
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def block_diag(*mats: np.ndarray) -> np.ndarray:
    """Small scipy-free block diagonal helper."""
    rows = sum(m.shape[0] for m in mats)
    cols = sum(m.shape[1] for m in mats)
    out = np.zeros((rows, cols), dtype=float)
    r = c = 0
    for m in mats:
        rr, cc = m.shape
        out[r:r + rr, c:c + cc] = m
        r += rr
        c += cc
    return out


# -----------------------------------------------------------------------------
# Measurement / coordinate manager
# -----------------------------------------------------------------------------

@dataclass
class SensorConfig:
    pos_ned: np.ndarray
    R: np.ndarray


class ExtendedCoordinateManager:
    """
    Computes h(x), H(x), and R for radar, camera and AIS.

    Radar and camera produce range-bearing directly.
    AIS produces absolute NED target position in the JSON, but for the project
    we convert it to an implied range-bearing measurement relative to vessel_pos.
    """

    def __init__(
        self,
        radar_pos: Iterable[float] = (0.0, 0.0),
        camera_pos: Iterable[float] = (-80.0, 120.0),
        sigma_radar_r: float = 5.0,
        sigma_radar_bearing: float = np.deg2rad(0.3),
        sigma_camera_r: float = 8.0,
        sigma_camera_bearing: float = np.deg2rad(0.15),
        sigma_ais_pos: float = 4.0,
        sigma_gnss_pos: float = 2.0,
    ):
        self.sensors: Dict[str, SensorConfig] = {
            "radar": SensorConfig(
                np.asarray(radar_pos, dtype=float),
                np.diag([sigma_radar_r**2, sigma_radar_bearing**2]),
            ),
            "camera": SensorConfig(
                np.asarray(camera_pos, dtype=float),
                np.diag([sigma_camera_r**2, sigma_camera_bearing**2]),
            ),
        }
        self.sigma_ais_pos = float(sigma_ais_pos)
        self.sigma_gnss_pos = float(sigma_gnss_pos)

    @staticmethod
    def h_range_bearing(x: np.ndarray, sensor_pos: np.ndarray) -> np.ndarray:
        dN = float(x[0] - sensor_pos[0])
        dE = float(x[1] - sensor_pos[1])
        r = math.hypot(dN, dE)
        bearing = math.atan2(dE, dN)
        return np.array([r, bearing], dtype=float)

    @staticmethod
    def H_range_bearing(x: np.ndarray, sensor_pos: np.ndarray) -> np.ndarray:
        dN = float(x[0] - sensor_pos[0])
        dE = float(x[1] - sensor_pos[1])
        q = max(dN*dN + dE*dE, 1e-9)
        r = math.sqrt(q)

        H = np.zeros((2, 4), dtype=float)
        H[0, 0] = dN / r
        H[0, 1] = dE / r
        H[1, 0] = -dE / q
        H[1, 1] = dN / q
        return H

    def sensor_position(self, sensor_id: str, vessel_pos: Optional[np.ndarray] = None) -> np.ndarray:
        sid = sensor_id.lower()
        if sid in ("radar", "camera"):
            return self.sensors[sid].pos_ned
        if sid == "ais":
            if vessel_pos is None:
                raise ValueError("AIS update needs vessel_pos from closest GNSS fix")
            return np.asarray(vessel_pos, dtype=float)
        raise ValueError(f"Unsupported sensor_id: {sensor_id}")

    def R_for_sensor(self, sensor_id: str, x_pred: Optional[np.ndarray] = None,
                     vessel_pos: Optional[np.ndarray] = None) -> np.ndarray:
        sid = sensor_id.lower()
        if sid in ("radar", "camera"):
            return self.sensors[sid].R

        if sid == "ais":
            # AIS and GNSS are position noises. After converting position to
            # range-bearing, approximate covariance using first-order propagation:
            # R_rb = J_pos * R_pos * J_pos^T.
            # If x_pred/vessel_pos not available, use a safe fallback.
            if x_pred is None or vessel_pos is None:
                return np.diag([self.sigma_ais_pos**2 + self.sigma_gnss_pos**2,
                                np.deg2rad(0.3)**2])

            sensor_pos = np.asarray(vessel_pos, dtype=float)
            dN = float(x_pred[0] - sensor_pos[0])
            dE = float(x_pred[1] - sensor_pos[1])
            q = max(dN*dN + dE*dE, 1e-9)
            r = math.sqrt(q)
            J = np.array([[dN / r, dE / r],
                          [-dE / q, dN / q]], dtype=float)
            R_pos = (self.sigma_ais_pos**2 + self.sigma_gnss_pos**2) * np.eye(2)
            return J @ R_pos @ J.T

        raise ValueError(f"Unsupported sensor_id: {sensor_id}")

    @staticmethod
    def ais_ned_to_range_bearing(meas: Dict[str, Any], vessel_pos: np.ndarray) -> np.ndarray:
        target_pos = np.array([float(meas["north_m"]), float(meas["east_m"])], dtype=float)
        d = target_pos - np.asarray(vessel_pos, dtype=float)
        return np.array([np.linalg.norm(d), math.atan2(d[1], d[0])], dtype=float)


# -----------------------------------------------------------------------------
# EKF core
# -----------------------------------------------------------------------------

class CVEKF:
    """Constant-velocity EKF for state [N,E,vN,vE]."""

    def __init__(self, x0: np.ndarray, P0: np.ndarray,
                 sigma_a: float = 0.05, t0: float = 0.0):
        self.x = np.asarray(x0, dtype=float).reshape(4)
        self.P = np.asarray(P0, dtype=float).reshape(4, 4)
        self.sigma_a = float(sigma_a)
        self.t = float(t0)
        self.nis_history: List[Tuple[float, str, float]] = []

    @staticmethod
    def F_Q(dt: float, sigma_a: float) -> Tuple[np.ndarray, np.ndarray]:
        dt = max(float(dt), 0.0)
        F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=float)
        q = sigma_a**2
        Q = q * np.array([
            [dt**4/4, 0, dt**3/2, 0],
            [0, dt**4/4, 0, dt**3/2],
            [dt**3/2, 0, dt**2, 0],
            [0, dt**3/2, 0, dt**2],
        ], dtype=float)
        return F, Q

    def predict_to(self, t: float) -> None:
        dt = float(t) - self.t
        if dt < -1e-9:
            # This simple implementation ignores out-of-sequence measurements.
            # For your T5 queue, sort measurements by time before processing.
            return
        F, Q = self.F_Q(dt, self.sigma_a)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        self.t = float(t)

    def update(self, z: np.ndarray, h: np.ndarray, H: np.ndarray, R: np.ndarray,
               sensor_id: str = "") -> float:
        z = np.asarray(z, dtype=float).reshape(-1)
        h = np.asarray(h, dtype=float).reshape(-1)
        y = z - h

        # Every measurement used here is [range, bearing] or stacked pairs.
        # Wrap every second component: 1, 3, 5, ...
        for k in range(1, len(y), 2):
            y[k] = wrap_angle(y[k])

        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y

        # Joseph form is more numerically stable than (I-KH)P.
        I = np.eye(4)
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ R @ K.T

        nis = float(y.T @ np.linalg.inv(S) @ y)
        self.nis_history.append((self.t, sensor_id, nis))
        return nis


# -----------------------------------------------------------------------------
# Extended tracker: T4 + T5
# -----------------------------------------------------------------------------

class ExtendedMultiSensorTracker:
    """
    Single-target extension for T4 and T5.

    It is meant to be called from your teammate's tracker.py. Later, when T6/T7
    are done, this same logic can be used inside each track object.
    """

    def __init__(self, coord: Optional[ExtendedCoordinateManager] = None,
                 sigma_a: float = 0.05):
        self.coord = coord or ExtendedCoordinateManager()
        self.sigma_a = float(sigma_a)
        self.ekf: Optional[CVEKF] = None
        self.history: List[Dict[str, Any]] = []

    def initialise_from_measurement(self, meas: Dict[str, Any],
                                    vessel_pos: Optional[np.ndarray] = None) -> None:
        sid = meas["sensor_id"].lower()
        t = float(meas["time"])

        if sid in ("radar", "camera"):
            sensor_pos = self.coord.sensor_position(sid)
            r = float(meas["range_m"])
            b = float(meas["bearing_rad"])
            pos = sensor_pos + np.array([r * math.cos(b), r * math.sin(b)])
        elif sid == "ais":
            pos = np.array([float(meas["north_m"]), float(meas["east_m"])], dtype=float)
        else:
            raise ValueError("Can only initialise from radar, camera or AIS")

        x0 = np.array([pos[0], pos[1], 0.0, 0.0], dtype=float)
        P0 = np.diag([50.0**2, 50.0**2, 5.0**2, 5.0**2])
        self.ekf = CVEKF(x0, P0, sigma_a=self.sigma_a, t0=t)
        self._save_history(t, f"init_{sid}")

    def update_one(self, meas: Dict[str, Any], vessel_pos: Optional[np.ndarray] = None) -> Optional[float]:
        sid = meas["sensor_id"].lower()
        if sid == "gnss":
            return None
        if sid not in ("radar", "camera", "ais"):
            return None
        if bool(meas.get("is_false_alarm", False)):
            # For T4/T5 single target validation, ignore known false alarms from simulator.
            # In T6 this must be replaced by gating + data association.
            return None

        if self.ekf is None:
            self.initialise_from_measurement(meas, vessel_pos=vessel_pos)
            return None

        t = float(meas["time"])
        self.ekf.predict_to(t)

        if sid in ("radar", "camera"):
            sensor_pos = self.coord.sensor_position(sid)
            z = np.array([float(meas["range_m"]), float(meas["bearing_rad"])], dtype=float)
        else:  # AIS
            if vessel_pos is None:
                raise ValueError("AIS measurement needs vessel_pos")
            sensor_pos = self.coord.sensor_position("ais", vessel_pos=vessel_pos)
            z = self.coord.ais_ned_to_range_bearing(meas, vessel_pos=sensor_pos)

        h = self.coord.h_range_bearing(self.ekf.x, sensor_pos)
        H = self.coord.H_range_bearing(self.ekf.x, sensor_pos)
        R = self.coord.R_for_sensor(sid, x_pred=self.ekf.x, vessel_pos=sensor_pos if sid == "ais" else None)
        nis = self.ekf.update(z, h, H, R, sensor_id=sid)
        self._save_history(t, sid)
        return nis

    def update_joint_radar_camera(self, radar_meas: Dict[str, Any], camera_meas: Dict[str, Any]) -> Optional[float]:
        """Optional T4 joint update if radar and camera are considered simultaneous."""
        if self.ekf is None:
            self.initialise_from_measurement(radar_meas)
            return None

        t = max(float(radar_meas["time"]), float(camera_meas["time"]))
        self.ekf.predict_to(t)

        z_parts, h_parts, H_parts, R_parts = [], [], [], []
        for meas in (radar_meas, camera_meas):
            sid = meas["sensor_id"].lower()
            sensor_pos = self.coord.sensor_position(sid)
            z_parts.append(np.array([float(meas["range_m"]), float(meas["bearing_rad"])]))
            h_parts.append(self.coord.h_range_bearing(self.ekf.x, sensor_pos))
            H_parts.append(self.coord.H_range_bearing(self.ekf.x, sensor_pos))
            R_parts.append(self.coord.R_for_sensor(sid))

        z = np.hstack(z_parts)
        h = np.hstack(h_parts)
        H = np.vstack(H_parts)
        R = block_diag(*R_parts)
        nis = self.ekf.update(z, h, H, R, sensor_id="joint_radar_camera")
        self._save_history(t, "joint_radar_camera")
        return nis

    def process_measurements_sequential(self, measurements: List[Dict[str, Any]],
                                        vessel_positions: Optional[np.ndarray] = None,
                                        allowed_sensors: Tuple[str, ...] = ("radar", "camera", "ais")) -> List[Dict[str, Any]]:
        """Process a time-sorted measurement list. This is the main T4/T5 loop."""
        measurements = sorted(measurements, key=lambda m: float(m["time"]))
        for meas in measurements:
            sid = meas["sensor_id"].lower()
            if sid not in allowed_sensors:
                continue
            vessel_pos = closest_vessel_position(float(meas["time"]), vessel_positions) if sid == "ais" else None
            self.update_one(meas, vessel_pos=vessel_pos)
        return self.history

    def _save_history(self, t: float, update_type: str) -> None:
        if self.ekf is None:
            return
        self.history.append({
            "time": float(t),
            "update": update_type,
            "x": self.ekf.x.copy(),
            "P": self.ekf.P.copy(),
        })


# -----------------------------------------------------------------------------
# JSON helpers for harbour_sim_output/scenario_*.json
# -----------------------------------------------------------------------------

def load_scenario_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def closest_vessel_position(t: float, vessel_positions: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """
    vessel_positions format from simulator JSON: [[time, N, E], ...]
    Returns [N, E] closest in time.
    """
    if vessel_positions is None:
        return None
    arr = np.asarray(vessel_positions, dtype=float)
    if arr.size == 0:
        return None
    idx = int(np.argmin(np.abs(arr[:, 0] - float(t))))
    return arr[idx, 1:3]


def run_extended_tracker_on_json(path: str,
                                 allowed_sensors: Tuple[str, ...] = ("radar", "camera", "ais")) -> ExtendedMultiSensorTracker:
    """Convenience function for quick testing from tracker.py or a notebook."""
    data = load_scenario_json(path)
    tracker = ExtendedMultiSensorTracker()
    vessel_positions = np.asarray(data.get("vessel_positions", []), dtype=float)
    tracker.process_measurements_sequential(
        data["measurements"],
        vessel_positions=vessel_positions,
        allowed_sensors=allowed_sensors,
    )
    return tracker


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run T4/T5 extended single-target tracker on scenario JSON")
    parser.add_argument("json_path", help="Path to harbour_sim_output/scenario_*.json")
    parser.add_argument("--sensors", nargs="+", default=["radar", "camera", "ais"],
                        help="Sensors to use, e.g. --sensors radar camera")
    args = parser.parse_args()

    trk = run_extended_tracker_on_json(args.json_path, allowed_sensors=tuple(args.sensors))
    if trk.ekf is None:
        print("No track was initialised.")
    else:
        print("Final state [N, E, vN, vE]:", np.round(trk.ekf.x, 3))
        print("Number of updates/history entries:", len(trk.history))
