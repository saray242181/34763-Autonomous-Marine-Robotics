"""
State:
    x = [p_N, p_E, v_N, v_E]^T

Bearing:
    bearing = atan2(delta_E, delta_N)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple, Any
import math
import numpy as np


def wrap_angle(angle: float | np.ndarray) -> float | np.ndarray:
    """Wrap angle(s) to [-pi, pi)."""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


@dataclass
class SensorConfig:
    """Fixed configuration of one sensor."""
    pos_ned: np.ndarray
    R: np.ndarray
    output_type: str


class CoordinateManager:
    """
    Coordinate Frame Manager for radar, stereo camera, AIS, and GNSS.

    Radar:
        Fixed at NED origin [0, 0].

    Camera:
        Fixed at default NED position [-80, 120] m.

    AIS:
        Raw AIS gives absolute target position [N, E].
        Here it is converted to range-bearing from the vessel.

    GNSS:
        Gives own-vessel position [N, E].
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
    ) -> None:
        self.sigma_ais_pos = float(sigma_ais_pos)
        self.sigma_gnss_pos = float(sigma_gnss_pos)

        self.sensors: Dict[str, SensorConfig] = {
            "radar": SensorConfig(
                pos_ned=np.asarray(radar_pos, dtype=float),
                R=np.diag([sigma_radar_r**2, sigma_radar_bearing**2]),
                output_type="range_bearing",
            ),
            "camera": SensorConfig(
                pos_ned=np.asarray(camera_pos, dtype=float),
                R=np.diag([sigma_camera_r**2, sigma_camera_bearing**2]),
                output_type="range_bearing",
            ),
            "gnss": SensorConfig(
                pos_ned=np.array([np.nan, np.nan], dtype=float),
                R=(sigma_gnss_pos**2) * np.eye(2),
                output_type="position",
            ),
        }

        self.vessel_pos_ned: Optional[np.ndarray] = None
        self.vessel_pos_time: Optional[float] = None

    def update_vessel_position(
        self,
        north_m: float,
        east_m: float,
        time_s: Optional[float] = None,
    ) -> None:
        """Update vessel position using GNSS."""
        self.vessel_pos_ned = np.array([north_m, east_m], dtype=float)

        if time_s is not None:
            self.vessel_pos_time = float(time_s)

    def update_vessel_position_from_measurement(self, meas: Dict[str, Any]) -> None:
        """Update vessel position directly from a GNSS measurement dictionary."""
        time_s = self._measurement_time(meas)

        self.update_vessel_position(
            north_m=float(meas["north_m"]),
            east_m=float(meas["east_m"]),
            time_s=time_s,
        )

    def sensor_position(
        self,
        sensor_id: str,
        vessel_pos: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Return sensor position [N, E] in NED frame.

        For AIS, the sensor position is the vessel position.
        """
        sensor_id = sensor_id.lower()

        if sensor_id in ("radar", "camera"):
            return self.sensors[sensor_id].pos_ned.copy()

        if sensor_id == "ais":
            if vessel_pos is not None:
                return np.asarray(vessel_pos, dtype=float)

            if self.vessel_pos_ned is not None:
                return self.vessel_pos_ned.copy()

            raise ValueError("AIS sensor position requires vessel position from GNSS.")

        if sensor_id == "gnss":
            raise ValueError("GNSS is not a target sensor. It provides vessel position.")

        raise ValueError(f"Unsupported sensor_id: {sensor_id}")

    def h(
        self,
        x: np.ndarray,
        sensor_id: str,
        vessel_pos: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        General measurement function.

        Returns:
            [range, bearing]
        """
        sensor_pos = self.sensor_position(sensor_id, vessel_pos=vessel_pos)
        return self.h_range_bearing(x, sensor_pos)

    def H(
        self,
        x: np.ndarray,
        sensor_id: str,
        vessel_pos: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        General measurement Jacobian.
        """
        sensor_pos = self.sensor_position(sensor_id, vessel_pos=vessel_pos)
        return self.H_range_bearing(x, sensor_pos)

    @staticmethod
    def h_range_bearing(
        x: np.ndarray,
        sensor_pos: np.ndarray,
    ) -> np.ndarray:
        """
        Range-bearing measurement model.

        h(x) = [
            sqrt((p_N - s_N)^2 + (p_E - s_E)^2),
            atan2(p_E - s_E, p_N - s_N)
        ]
        """
        x = np.asarray(x, dtype=float).reshape(-1)
        sensor_pos = np.asarray(sensor_pos, dtype=float).reshape(2)

        dN = float(x[0] - sensor_pos[0])
        dE = float(x[1] - sensor_pos[1])

        range_m = math.hypot(dN, dE)
        bearing_rad = math.atan2(dE, dN)

        return np.array([range_m, bearing_rad], dtype=float)

    @staticmethod
    def H_range_bearing(
        x: np.ndarray,
        sensor_pos: np.ndarray,
    ) -> np.ndarray:
        """
        Jacobian of range-bearing model.

        State:
            x = [p_N, p_E, v_N, v_E]
        """
        x = np.asarray(x, dtype=float).reshape(-1)
        sensor_pos = np.asarray(sensor_pos, dtype=float).reshape(2)

        dN = float(x[0] - sensor_pos[0])
        dE = float(x[1] - sensor_pos[1])

        q = max(dN**2 + dE**2, 1e-9)
        r = math.sqrt(q)

        H = np.zeros((2, 4), dtype=float)

        H[0, 0] = dN / r
        H[0, 1] = dE / r

        H[1, 0] = -dE / q
        H[1, 1] = dN / q

        return H

    def measurement_to_z(
        self,
        meas: Dict[str, Any],
        vessel_pos: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, str, np.ndarray]:
        """
        Convert measurement dictionary to EKF measurement z.

        Returns:
            z, sensor_id, sensor_pos
        """
        sensor_id = self._sensor_id(meas)

        if sensor_id in ("radar", "camera"):
            sensor_pos = self.sensor_position(sensor_id)

            z = np.array(
                [
                    float(meas["range_m"]),
                    float(meas["bearing_rad"]),
                ],
                dtype=float,
            )

            z[1] = wrap_angle(z[1])

            return z, sensor_id, sensor_pos

        if sensor_id == "ais":
            sensor_pos = self.sensor_position("ais", vessel_pos=vessel_pos)
            z = self.ais_position_to_range_bearing(meas, vessel_pos=sensor_pos)

            return z, sensor_id, sensor_pos

        raise ValueError(f"Unsupported measurement sensor: {sensor_id}")

    @staticmethod
    def ais_position_to_range_bearing(
        meas: Dict[str, Any],
        vessel_pos: np.ndarray,
    ) -> np.ndarray:
        """
        Convert AIS absolute NED target position into range-bearing from vessel.
        """
        vessel_pos = np.asarray(vessel_pos, dtype=float).reshape(2)

        target_pos = np.array(
            [
                float(meas["north_m"]),
                float(meas["east_m"]),
            ],
            dtype=float,
        )

        delta = target_pos - vessel_pos

        range_m = float(np.linalg.norm(delta))
        bearing_rad = math.atan2(float(delta[1]), float(delta[0]))

        return np.array([range_m, wrap_angle(bearing_rad)], dtype=float)

    def R(
        self,
        sensor_id: str,
        x_pred: Optional[np.ndarray] = None,
        vessel_pos: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Return measurement covariance matrix.
        """
        sensor_id = sensor_id.lower()

        if sensor_id in ("radar", "camera"):
            return self.sensors[sensor_id].R.copy()

        if sensor_id == "ais":
            return self.R_ais_range_bearing(
                x_pred=x_pred,
                vessel_pos=vessel_pos,
            )

        if sensor_id == "gnss":
            return self.sensors["gnss"].R.copy()

        raise ValueError(f"Unsupported sensor_id: {sensor_id}")

    def R_ais_range_bearing(
        self,
        x_pred: Optional[np.ndarray],
        vessel_pos: Optional[np.ndarray],
    ) -> np.ndarray:
        """
        Approximate AIS covariance after converting NED position to range-bearing.

        Total position noise:
            AIS target position noise + GNSS vessel position noise
        """
        total_position_variance = self.sigma_ais_pos**2 + self.sigma_gnss_pos**2

        if x_pred is None or vessel_pos is None:
            return np.diag(
                [
                    total_position_variance,
                    np.deg2rad(0.3) ** 2,
                ]
            )

        x_pred = np.asarray(x_pred, dtype=float).reshape(-1)
        vessel_pos = np.asarray(vessel_pos, dtype=float).reshape(2)

        dN = float(x_pred[0] - vessel_pos[0])
        dE = float(x_pred[1] - vessel_pos[1])

        q = max(dN**2 + dE**2, 1e-9)
        r = math.sqrt(q)

        J_pos_to_rb = np.array(
            [
                [dN / r, dE / r],
                [-dE / q, dN / q],
            ],
            dtype=float,
        )

        R_pos = total_position_variance * np.eye(2)

        return J_pos_to_rb @ R_pos @ J_pos_to_rb.T

    def ned_from_range_bearing(
        self,
        range_m: float,
        bearing_rad: float,
        sensor_id: str,
        vessel_pos: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Convert range-bearing into absolute NED position.
        Useful for track initialization.
        """
        sensor_pos = self.sensor_position(sensor_id, vessel_pos=vessel_pos)

        return sensor_pos + np.array(
            [
                range_m * math.cos(bearing_rad),
                range_m * math.sin(bearing_rad),
            ],
            dtype=float,
        )

    def expected_measurement_for_position(
        self,
        target_pos_ned: Iterable[float],
        sensor_id: str,
        vessel_pos: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Compute expected [range, bearing] for a target position [N, E].
        Useful for unit tests.
        """
        target_pos_ned = np.asarray(target_pos_ned, dtype=float).reshape(2)

        x = np.array(
            [
                target_pos_ned[0],
                target_pos_ned[1],
                0.0,
                0.0,
            ]
        )

        return self.h(x, sensor_id=sensor_id, vessel_pos=vessel_pos)

    @staticmethod
    def _sensor_id(meas: Dict[str, Any]) -> str:
        return str(meas.get("sensor_id", meas.get("sensor", ""))).lower()

    @staticmethod
    def _measurement_time(meas: Dict[str, Any]) -> Optional[float]:
        for key in ("time", "timestamp", "t"):
            if key in meas:
                return float(meas[key])
        return None


if __name__ == "__main__":
    cm = CoordinateManager()

    x = np.array([100.0, 200.0, 1.0, -1.0])

    print("Radar h:", cm.h(x, "radar"))
    print("Radar H:\n", cm.H(x, "radar"))

    print("Camera h:", cm.h(x, "camera"))
    print("Camera H:\n", cm.H(x, "camera"))

    cm.update_vessel_position(20.0, 30.0, time_s=10.0)

    ais_meas = {
        "sensor_id": "ais",
        "time": 10.0,
        "north_m": 100.0,
        "east_m": 200.0,
    }

    z_ais, sid, ais_sensor_pos = cm.measurement_to_z(ais_meas)

    print("AIS converted z:", z_ais)
    print("AIS sensor position:", ais_sensor_pos)
    print("AIS R:\n", cm.R("ais", x_pred=x, vessel_pos=ais_sensor_pos))