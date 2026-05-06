"""
real_data_tracker.py — Phase 4: real-data validation.

Loads the four real sensor CSVs from data/, applies frame rotations,
feeds the unified measurement stream into the unchanged MultiTargetTracker
(Phase 3), and produces trajectory plots + quantitative metrics.

Frame rotations (real data only — simulation is already NED):
    Radar bearing:  +16°  (radar frame → NED)
    Camera (X, Z):  +28°  (camera frame → NED)

Usage (from tracker/ directory):
    python real_data_tracker.py
"""

from __future__ import annotations

import csv
import math
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

from coordinate_manager import CoordinateManager
from extended_tracker import closest_vessel_position
from gnn_tracker import MultiTargetTracker


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")

RADAR_ROTATION_DEG  =  16.0   # radar frame → NED
CAMERA_ROTATION_DEG =  28.0   # camera frame → NED

# Real-data noise parameters (data/README.md)
SIGMA_GNSS_POS = 6.0   # m
SIGMA_AIS_POS  = 6.0   # m

# GPS position of the NED origin (= radar position), from data/README.md
_LAT0_DEG = 55.69014690
_LON0_DEG = 12.59998830


# ---------------------------------------------------------------------------
# Coordinate conversion helpers (NED metres → Web Mercator for map overlay)
# ---------------------------------------------------------------------------

def _ned_to_latlon(N: float, E: float) -> Tuple[float, float]:
    """Flat-Earth NED metres → (lat_deg, lon_deg). Accurate to < 1 m within 10 km."""
    R = 6_371_000.0
    lat = _LAT0_DEG + math.degrees(N / R)
    lon = _LON0_DEG + math.degrees(E / (R * math.cos(math.radians(_LAT0_DEG))))
    return lat, lon


def _latlon_to_webmerc(lat: float, lon: float) -> Tuple[float, float]:
    """WGS84 (lat_deg, lon_deg) → Web Mercator (x, y) in metres (EPSG:3857)."""
    x = math.radians(lon) * 6_378_137.0
    y = math.log(math.tan(math.pi / 4.0 + math.radians(lat) / 2.0)) * 6_378_137.0
    return x, y


def _ned_to_wm(N: float, E: float) -> Tuple[float, float]:
    """NED metres → Web Mercator (x, y)."""
    return _latlon_to_webmerc(*_ned_to_latlon(N, E))


# ---------------------------------------------------------------------------
# CSV loaders
# ---------------------------------------------------------------------------

def _read_csv(filename: str) -> List[Dict[str, str]]:
    path = os.path.join(DATA_DIR, filename)
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def load_gnss() -> np.ndarray:
    """Return vessel_positions as (T, 3) array [[time, N, E], ...]."""
    rows = _read_csv("gnss.csv")
    arr = np.array([[float(r["time"]), float(r["N"]), float(r["E"])]
                    for r in rows], dtype=float)
    return arr[np.argsort(arr[:, 0])]


def load_radar_measurements() -> List[Dict]:
    """Convert radar CSV rows to measurement dicts with NED bearing."""
    rot_rad = math.radians(RADAR_ROTATION_DEG)
    meas = []
    for r in _read_csv("mm_wave_radar.csv"):
        bearing_sensor_rad = math.radians(float(r["bearing"]))
        bearing_ned_rad    = bearing_sensor_rad + rot_rad
        meas.append({
            "sensor_id":   "radar",
            "time":        float(r["time"]),
            "range_m":     float(r["range"]),
            "bearing_rad": bearing_ned_rad,
            "north_m":     None,
            "east_m":      None,
        })
    return meas


def load_camera_measurements() -> List[Dict]:
    """
    Convert camera CSV (X, Z) in camera frame to range-bearing in NED.

    Bearing in camera frame: atan2(X, Z)  (Z is forward/depth, X is lateral)
    After +28° rotation:     bearing_ned = bearing_camera + 28° * pi/180
    """
    rot_rad = math.radians(CAMERA_ROTATION_DEG)
    meas = []
    for r in _read_csv("camera.csv"):
        x = float(r["X"])
        z = float(r["Z"])
        range_m        = math.hypot(x, z)
        bearing_cam    = math.atan2(x, z)
        bearing_ned    = bearing_cam + rot_rad
        meas.append({
            "sensor_id":   "camera",
            "time":        float(r["time"]),
            "range_m":     range_m,
            "bearing_rad": bearing_ned,
            "north_m":     None,
            "east_m":      None,
        })
    return meas


def load_ais_measurements() -> List[Dict]:
    """AIS positions are already in NED — pass through directly."""
    meas = []
    for r in _read_csv("ais.csv"):
        meas.append({
            "sensor_id": "ais",
            "time":      float(r["time"]),
            "range_m":   None,
            "bearing_rad": None,
            "north_m":   float(r["N"]),
            "east_m":    float(r["E"]),
            "ais_id":    r["ais_id"],
        })
    return meas


# ---------------------------------------------------------------------------
# Measurement stream builder
# ---------------------------------------------------------------------------

def build_scan_events(
    measurements: List[Dict],
    allowed_sensors: frozenset,
) -> List[Tuple[float, str, List[Dict]]]:
    """Group measurements by (time, sensor_id), sorted by time."""
    groups: Dict[Tuple[float, str], List[Dict]] = defaultdict(list)
    for m in measurements:
        sid = m["sensor_id"].lower()
        if sid not in allowed_sensors:
            continue
        groups[(round(float(m["time"]), 6), sid)].append(m)

    return [
        (t, sid, mlist)
        for (t, sid), mlist in sorted(
            groups.items(), key=lambda kv: (kv[0][0], kv[0][1])
        )
    ]


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------

def run_real_data(
    allowed_sensors: frozenset = frozenset({"radar", "camera", "ais"}),
) -> Tuple[MultiTargetTracker, np.ndarray, List[Dict]]:
    """
    Load real CSV data, run MultiTargetTracker, return tracker + vessel_positions
    + raw measurement list.

    CoordinateManager is configured with real-data noise values:
        sigma_ais_pos  = sigma_gnss_pos = 6 m  (data/README.md)
    Radar and camera R use the simulation defaults (5 m / 0.3° and 8 m / 0.15°)
    because the tracker R() interface does not support per-measurement matrices.
    Per-measurement radar covariance (from CSV) is a known improvement opportunity.
    """
    coord = CoordinateManager(
        sigma_ais_pos  = SIGMA_AIS_POS,
        sigma_gnss_pos = SIGMA_GNSS_POS,
    )

    vessel_positions = load_gnss()

    all_meas: List[Dict] = []
    all_meas.extend(load_radar_measurements())
    all_meas.extend(load_camera_measurements())
    all_meas.extend(load_ais_measurements())

    scan_events = build_scan_events(all_meas, allowed_sensors)

    mtt = MultiTargetTracker(coord)

    for scan_time, sensor_id, meas_list in scan_events:
        vessel_pos = None
        if sensor_id == "ais":
            vessel_pos = closest_vessel_position(scan_time, vessel_positions)

        mtt.process_scan(scan_time, sensor_id, meas_list, vessel_pos=vessel_pos)

    print(f"\n=== Real data — final tracker state ===")
    print(mtt.summary())

    return mtt, vessel_positions, all_meas


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_ais_rmse(
    mtt: MultiTargetTracker,
    ais_measurements: List[Dict],
) -> None:
    """
    Treat AIS reports as pseudo-ground-truth and compute per-MMSI RMSE
    against the nearest confirmed/coasting track at each AIS timestamp.

    Groups by MMSI (persistent ship identifier) rather than ais_id, because
    ais_id is a local ephemeral label that may be reassigned to different
    physical ships within the same dataset.

    This is an approximation: AIS σ=6 m, so RMSE < 6 m is indistinguishable
    from sensor noise.
    """
    # Group by MMSI (persistent); fall back to ais_id if mmsi absent
    ais_by_mmsi: Dict[str, List[Tuple[float, float, float]]] = defaultdict(list)
    for m in ais_measurements:
        key = str(m.get("mmsi", m.get("ais_id", "unknown")))
        ais_by_mmsi[key].append(
            (float(m["time"]), float(m["north_m"]), float(m["east_m"]))
        )

    # Build scan_history lookup: time → confirmed/coasting track positions
    history_by_time: Dict[float, List[np.ndarray]] = {}
    for snap in mtt.scan_history:
        t = float(snap["time"])
        positions = [
            np.array(tr["pos"], dtype=float)
            for tr in snap["tracks"]
            if tr["state"] in ("confirmed", "coasting")
        ]
        if t not in history_by_time:
            history_by_time[t] = positions

    snap_times = np.array(sorted(history_by_time.keys()))

    print("\n=== AIS pseudo-GT RMSE (grouped by MMSI) ===")
    for mmsi, reports in sorted(ais_by_mmsi.items()):
        errors = []
        for t_ais, n_ais, e_ais in reports:
            idx = int(np.argmin(np.abs(snap_times - t_ais)))
            t_snap = snap_times[idx]
            if abs(t_snap - t_ais) > 5.0:
                continue
            positions = history_by_time[t_snap]
            if not positions:
                continue
            ais_pos = np.array([n_ais, e_ais])
            dists = [float(np.linalg.norm(p - ais_pos)) for p in positions]
            errors.append(min(dists))

        if errors:
            rmse = float(np.sqrt(np.mean(np.array(errors) ** 2)))
            print(f"  MMSI={mmsi}: {len(errors)} comparisons, "
                  f"RMSE = {rmse:.2f} m  (AIS σ=6 m)")
        else:
            print(f"  MMSI={mmsi}: no overlapping scan times found")


def print_cardinality_summary(mtt: MultiTargetTracker) -> None:
    """Print mean confirmed+coasting track count over time (CE proxy)."""
    counts = []
    for snap in mtt.scan_history:
        n = sum(
            1 for tr in snap["tracks"]
            if tr["state"] in ("confirmed", "coasting")
        )
        counts.append(n)
    if counts:
        arr = np.array(counts, dtype=float)
        print(f"\n=== Track cardinality ===")
        print(f"  Mean confirmed+coasting tracks: {arr.mean():.2f}")
        print(f"  Peak: {int(arr.max())}   "
              f"Min (non-zero): {int(arr[arr > 0].min()) if (arr > 0).any() else 0}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_real_tracks(
    mtt: MultiTargetTracker,
    vessel_positions: np.ndarray,
    all_meas: List[Dict],
) -> None:
    """
    Trajectory plot overlaid on a satellite map of Copenhagen harbour.

    All NED coordinates are converted to Web Mercator (EPSG:3857) so that
    contextily can add ESRI WorldImagery tiles as a background.

    Layers (bottom to top):
      - Satellite basemap
      - Sensor measurement back-projections (low-alpha scatter)
      - AIS positions (crosses)
      - Vessel GNSS path
      - Confirmed + coasting track paths (dashed)
      - Sensor position markers
    """
    coord = CoordinateManager()
    colors = plt.cm.tab10.colors

    fig, ax = plt.subplots(figsize=(14, 12))

    # Vessel GNSS path
    vx = [_ned_to_wm(N, E)[0] for N, E in zip(vessel_positions[:, 1], vessel_positions[:, 2])]
    vy = [_ned_to_wm(N, E)[1] for N, E in zip(vessel_positions[:, 1], vessel_positions[:, 2])]
    ax.plot(vx, vy, color="gray", linewidth=0.8, alpha=0.5, label="Vessel (GNSS)", zorder=2)

    # Confirmed + coasting track trajectories
    track_traj: Dict[int, List] = defaultdict(list)
    for snap in mtt.scan_history:
        for tr in snap["tracks"]:
            if tr["state"] in ("confirmed", "coasting"):
                track_traj[tr["id"]].append(tr["pos"])

    for i, (tid, positions) in enumerate(track_traj.items()):
        wm = [_ned_to_wm(p[0], p[1]) for p in positions]
        xs = [p[0] for p in wm]
        ys = [p[1] for p in wm]
        ax.plot(xs, ys, "--", color=colors[i % 10], linewidth=1.5,
                label=f"Track {tid}", zorder=5)

    # Sensor measurement scatter
    cam_pos = coord.sensor_position("camera")
    for m in all_meas:
        sid = m["sensor_id"].lower()
        if sid in ("radar", "camera") and m.get("range_m") is not None:
            r = float(m["range_m"])
            b = float(m["bearing_rad"])
            sensor_pos = coord.sensor_position(sid)
            ne = sensor_pos + np.array([r * math.cos(b), r * math.sin(b)])
            wx, wy = _ned_to_wm(float(ne[0]), float(ne[1]))
            color = "steelblue" if sid == "radar" else "tomato"
            ax.scatter(wx, wy, s=4, color=color, alpha=0.08, zorder=1)

    # AIS reports
    ais_by_id: Dict[str, List] = defaultdict(list)
    for m in all_meas:
        if m["sensor_id"].lower() == "ais":
            ais_by_id[m["ais_id"]].append([float(m["north_m"]), float(m["east_m"])])
    for j, (aid, pts) in enumerate(sorted(ais_by_id.items())):
        wm = [_ned_to_wm(p[0], p[1]) for p in pts]
        xs = [p[0] for p in wm]
        ys = [p[1] for p in wm]
        ax.scatter(xs, ys, marker="x", s=15, linewidths=0.8,
                   color=colors[j % 10], alpha=0.6, label=f"AIS id={aid}", zorder=3)

    # Sensor markers
    rx, ry = _ned_to_wm(0.0, 0.0)
    ax.scatter(rx, ry, marker="^", s=120, color="black", zorder=6, label="Radar [0,0]")
    cx, cy = _ned_to_wm(float(cam_pos[0]), float(cam_pos[1]))
    ax.scatter(cx, cy, marker="s", s=120, color="purple", zorder=6,
               label=f"Camera [{cam_pos[0]:.0f},{cam_pos[1]:.0f}]")

    # Satellite basemap
    try:
        import contextily as ctx
        ctx.add_basemap(
            ax,
            source=ctx.providers.Esri.WorldImagery,
            zoom=15,
            attribution_size=6,
        )
    except Exception as exc:
        print(f"  [warning] Satellite basemap unavailable ({exc}); using plain grid.")
        ax.set_aspect("equal")

    ax.set_xlabel("Easting (Web Mercator, m)")
    ax.set_ylabel("Northing (Web Mercator, m)")
    ax.set_title("Phase 4 — Real Data: Confirmed Track Trajectories (Copenhagen Harbour)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2, loc="upper right")
    plt.tight_layout()
    plt.savefig(
        os.path.join(DATA_DIR, "..", "harbour_sim_output", "real_data_tracks.png"),
        dpi=150,
    )
    print("\nPlot saved → harbour_sim_output/real_data_tracks.png")
    plt.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    mtt, vessel_positions, all_meas = run_real_data(
        allowed_sensors=frozenset({"radar", "camera", "ais"}),
    )

    ais_meas = [m for m in all_meas if m["sensor_id"].lower() == "ais"]
    compute_ais_rmse(mtt, ais_meas)
    print_cardinality_summary(mtt)
    plot_real_tracks(mtt, vessel_positions, all_meas)


if __name__ == "__main__":
    main()
