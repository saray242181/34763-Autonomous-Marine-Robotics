"""
multi_tracker.py — Multi-target scenario runner for Scenarios D and E.

Drives MultiTargetTracker (T6/T7) against the harbour simulation JSON files
and evaluates performance with MOTP + Cardinality Error (T8).

Usage (from tracker/ directory):
    python multi_tracker.py
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Dict, FrozenSet, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

from coordinate_manager import CoordinateManager
from extended_tracker import closest_vessel_position
from gnn_tracker import MultiTargetTracker
from metrics import evaluate_tracker_performance


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_scenario(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def build_scan_events(
    measurements: List[dict],
    allowed_sensors: FrozenSet[str],
) -> List[Tuple[float, str, List[dict]]]:
    """
    Group measurements by (time, sensor_id) and return a time-sorted list.

    GNSS entries are skipped — vessel positions are read from the dedicated
    vessel_positions array, not from the measurement stream.

    False alarms are NOT filtered here; MultiTargetTracker handles them via
    Mahalanobis gating internally.
    """
    groups: Dict[Tuple[float, str], List[dict]] = defaultdict(list)

    for m in measurements:
        sid = m["sensor_id"].lower()
        if sid == "gnss":
            continue
        if sid not in allowed_sensors:
            continue
        groups[(float(m["time"]), sid)].append(m)

    return [
        (t, sid, meas_list)
        for (t, sid), meas_list in sorted(
            groups.items(), key=lambda kv: (kv[0][0], kv[0][1])
        )
    ]


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------

def run_scenario(
    json_path: str,
    scenario_name: str = "D",
    allowed_sensors: Optional[FrozenSet[str]] = None,
) -> Tuple[MultiTargetTracker, dict]:
    """
    Load a scenario JSON, drive MultiTargetTracker through all scan events,
    and evaluate MOTP + CE.

    Returns the populated MultiTargetTracker and the raw scenario data dict.
    """
    if allowed_sensors is None:
        allowed_sensors = frozenset({"radar", "camera"})

    data = load_scenario(json_path)
    measurements = data["measurements"]
    vessel_positions = np.asarray(data.get("vessel_positions", []), dtype=float)

    coord = CoordinateManager()
    mtt = MultiTargetTracker(coord)

    scan_events = build_scan_events(measurements, allowed_sensors)

    for scan_time, sensor_id, meas_list in scan_events:
        vessel_pos = None
        if sensor_id == "ais":
            vessel_pos = closest_vessel_position(scan_time, vessel_positions)

        mtt.process_scan(
            scan_time,
            sensor_id,
            meas_list,
            vessel_pos=vessel_pos,
        )

    print(f"\n=== Scenario {scenario_name} — final tracker state ===")
    print(mtt.summary())

    motp, ce = evaluate_tracker_performance(data["ground_truth"], mtt.scan_history, scenario_name=scenario_name)
    print(f"\nScenario {scenario_name}  |  MOTP: {motp:.2f} m  |  Mean CE: {ce:.3f}")

    return mtt, data


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_multi_target_ned(
    data: dict,
    mtt: MultiTargetTracker,
    scenario_name: str = "D",
) -> None:
    """2-D NED scene: GT trajectories, confirmed track paths, sensor positions."""
    coord = CoordinateManager()
    colors = plt.cm.tab10.colors

    fig, ax = plt.subplots(figsize=(12, 10))

    # Ground-truth trajectories
    gt = data["ground_truth"]
    for idx, (tid, arr) in enumerate(gt.items()):
        arr = np.asarray(arr, dtype=float)
        valid = ~np.isnan(arr[:, 1])
        ax.plot(
            arr[valid, 2], arr[valid, 1],
            color=colors[idx % 10], linewidth=2.0,
            label=f"GT target {tid}", zorder=3,
        )

    # Confirmed track trajectories reconstructed from scan_history
    track_traj: Dict[int, list] = defaultdict(list)
    for snap in mtt.scan_history:
        for tr in snap["tracks"]:
            if tr["state"] == "confirmed":
                track_traj[tr["id"]].append(tr["pos"])

    for i, (tid, positions) in enumerate(track_traj.items()):
        pts = np.array(positions)
        ax.plot(
            pts[:, 1], pts[:, 0],
            "--", color=colors[i % 10], linewidth=1.2,
            label=f"Track {tid}", zorder=4,
        )

    # Sensor measurement back-projections (low alpha)
    for m in data["measurements"]:
        sid = m["sensor_id"].lower()
        if sid in ("radar", "camera") and m.get("range_m") is not None:
            sensor_pos = coord.sensor_position(sid)
            r = float(m["range_m"])
            b = float(m["bearing_rad"])
            ne = sensor_pos + np.array([r * np.cos(b), r * np.sin(b)])
            color = "steelblue" if sid == "radar" else "tomato"
            ax.scatter(ne[1], ne[0], s=6, color=color, alpha=0.15, zorder=1)

    # Sensor position markers
    ax.scatter(0, 0, marker="^", s=100, color="black", zorder=5, label="Radar [0,0]")
    cam_pos = coord.sensor_position("camera")
    ax.scatter(
        cam_pos[1], cam_pos[0],
        marker="s", s=100, color="purple", zorder=5,
        label=f"Camera [{cam_pos[0]:.0f},{cam_pos[1]:.0f}]",
    )

    ax.set_xlabel("East [m]")
    ax.set_ylabel("North [m]")
    ax.set_title(f"Scenario {scenario_name} — Multi-Target 2-D NED")
    ax.axis("equal")
    ax.grid(True)
    ax.legend(fontsize=8, ncol=2, loc="upper right")
    plt.tight_layout()
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "harbour_sim_output")
    fig.savefig(os.path.join(out_dir, f"scenario_{scenario_name}_tracks.png"), dpi=150)
    print(f"Track plot saved → harbour_sim_output/scenario_{scenario_name}_tracks.png")
    plt.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "harbour_sim_output")

    # Scenario D: 4 targets, radar + camera, 120 s, no AIS
    mtt_d, data_d = run_scenario(
        json_path=os.path.join(base, "scenario_D.json"),
        scenario_name="D",
        allowed_sensors=frozenset({"radar", "camera"}),
    )
    plot_multi_target_ned(data_d, mtt_d, scenario_name="D")

    # Scenario E: 6 targets, radar + camera + AIS, 180 s
    mtt_e, data_e = run_scenario(
        json_path=os.path.join(base, "scenario_E.json"),
        scenario_name="E",
        allowed_sensors=frozenset({"radar", "camera", "ais"}),
    )
    plot_multi_target_ned(data_e, mtt_e, scenario_name="E")


if __name__ == "__main__":
    main()
