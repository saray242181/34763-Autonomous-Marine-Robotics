import json
import numpy as np
import matplotlib.pyplot as plt

from ekf import EKF
from coordinate_manager import CoordinateManager
from extended_tracker import ExtendedTracker, initial_state_from_measurement


# -------------------------------------------------------------------------
# Basic helpers
# -------------------------------------------------------------------------

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def is_false_alarm(meas):
    return (
        meas.get("is_false_alarm", False)
        or int(meas.get("true_target_id", meas.get("target_id", 0))) == -1
    )


def first_valid_measurement(measurements, allowed_sensors):
    for m in sorted(measurements, key=lambda x: x["time"]):
        sid = m["sensor_id"].lower()

        if sid not in allowed_sensors:
            continue

        if is_false_alarm(m):
            continue

        return m

    raise RuntimeError("No valid measurement found.")


def extract_estimates(history):
    times = []
    est_N = []
    est_E = []

    for h in history:
        x = np.asarray(h["x"]).reshape(-1)
        times.append(h["time"])
        est_N.append(x[0])
        est_E.append(x[1])

    return np.array(times), np.array(est_N), np.array(est_E)


def get_first_ground_truth(data):
    gt = data["ground_truth"]
    target_id = list(gt.keys())[0]
    truth = np.asarray(gt[target_id], dtype=float)

    gt_time = truth[:, 0]
    gt_N = truth[:, 1]
    gt_E = truth[:, 2]

    valid = ~np.isnan(gt_N)

    return gt_time, gt_N, gt_E, valid, target_id


def compute_error_series(data, history):
    times, est_N, est_E = extract_estimates(history)
    gt_time, gt_N, gt_E, valid, _ = get_first_ground_truth(data)

    gt_N_interp = np.interp(times, gt_time[valid], gt_N[valid])
    gt_E_interp = np.interp(times, gt_time[valid], gt_E[valid])

    error_N = est_N - gt_N_interp
    error_E = est_E - gt_E_interp

    position_error = np.sqrt(error_N**2 + error_E**2)
    rmse = np.sqrt(np.mean(position_error**2))

    return times, position_error, rmse


def compute_rmse(data, history):
    _, _, rmse = compute_error_series(data, history)
    return rmse


def compute_rmse_in_window(data, history, t_start, t_end):
    times, position_error, _ = compute_error_series(data, history)

    mask = (times >= t_start) & (times <= t_end)

    if np.sum(mask) == 0:
        return np.nan

    return np.sqrt(np.mean(position_error[mask] ** 2))


def compute_update_count_in_window(history, t_start, t_end):
    return sum(t_start <= h["time"] <= t_end for h in history)


def print_pass_fail(name, passed, value_text=""):
    symbol = "PASS" if passed else "FAIL"
    print(f"{symbol}: {name} {value_text}")


# -------------------------------------------------------------------------
# NIS
# -------------------------------------------------------------------------

def nis_bounds_for_sensor(sensor_name):
    """
    2 DOF:
        radar, camera, ais range-bearing updates

    4 DOF:
        joint radar-camera update:
        z = [r_radar, bearing_radar, r_camera, bearing_camera]
    """
    if sensor_name == "joint_radar_camera":
        return 0.711, 9.488, 4

    return 0.103, 5.991, 2


def evaluate_nis_consistency(tracker_ext):
    if len(tracker_ext.nis_history) == 0:
        return {
            "available": False,
            "percentage_inside": np.nan,
            "mean_nis": np.nan,
            "n": 0,
            "by_sensor": {},
        }

    items = tracker_ext.nis_history

    inside_all = []
    values_all = []
    by_sensor = {}

    for item in items:
        sensor = item["sensor"]
        nis = float(item["nis"])

        lower, upper, dof = nis_bounds_for_sensor(sensor)

        inside = lower <= nis <= upper

        inside_all.append(inside)
        values_all.append(nis)

        if sensor not in by_sensor:
            by_sensor[sensor] = {
                "values": [],
                "inside": [],
                "lower": lower,
                "upper": upper,
                "dof": dof,
            }

        by_sensor[sensor]["values"].append(nis)
        by_sensor[sensor]["inside"].append(inside)

    by_sensor_out = {}

    for sensor, d in by_sensor.items():
        vals = np.asarray(d["values"])
        inside = np.asarray(d["inside"])

        by_sensor_out[sensor] = {
            "n": len(vals),
            "mean_nis": float(np.mean(vals)),
            "percentage_inside": 100.0 * np.mean(inside),
            "lower": d["lower"],
            "upper": d["upper"],
            "dof": d["dof"],
        }

    return {
        "available": True,
        "percentage_inside": 100.0 * np.mean(inside_all),
        "mean_nis": float(np.mean(values_all)),
        "n": len(values_all),
        "by_sensor": by_sensor_out,
    }


def print_nis_consistency(tracker_ext):
    result = evaluate_nis_consistency(tracker_ext)

    print("\nNIS CONSISTENCY TEST")
    print("-" * 40)

    if not result["available"]:
        print("No NIS values available.")
        return result

    print(f"Number of NIS samples: {result['n']}")
    print(f"Mean NIS: {result['mean_nis']:.3f}")
    print(f"Percentage inside correct chi-square bounds: {result['percentage_inside']:.1f}%")

    print_pass_fail(
        "> 90% NIS inside bounds",
        result["percentage_inside"] >= 90.0,
        f"({result['percentage_inside']:.1f}%)",
    )

    print("\nPer-sensor / per-update NIS:")
    for sensor, stats in result["by_sensor"].items():
        print(
            f"- {sensor}: "
            f"{stats['percentage_inside']:.1f}% inside, "
            f"mean NIS = {stats['mean_nis']:.2f}, "
            f"DOF = {stats['dof']}, "
            f"bounds = [{stats['lower']:.3f}, {stats['upper']:.3f}], "
            f"n = {stats['n']}"
        )

    return result


def plot_nis(tracker_ext):
    if len(tracker_ext.nis_history) == 0:
        print("No NIS values to plot.")
        return

    plt.figure(figsize=(9, 4))

    sensors = sorted(set(item["sensor"] for item in tracker_ext.nis_history))

    for sensor in sensors:
        times = []
        values = []

        for item in tracker_ext.nis_history:
            if item["sensor"] == sensor:
                times.append(item["time"])
                values.append(item["nis"])

        lower, upper, dof = nis_bounds_for_sensor(sensor)

        plt.scatter(times, values, s=15, label=f"{sensor} ({dof} DOF)")

    plt.axhline(5.991, linestyle="--", label="2 DOF upper 95%")
    plt.axhline(9.488, linestyle="--", label="4 DOF upper 95%")

    plt.xlabel("Time [s]")
    plt.ylabel("NIS")
    plt.title("NIS consistency")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# -------------------------------------------------------------------------
# Plotting
# -------------------------------------------------------------------------

def plot_harbour_debug(data, history):
    measurements = data["measurements"]

    times, est_N, est_E = extract_estimates(history)
    gt_time, gt_N, gt_E, valid, target_id = get_first_ground_truth(data)
    err_times, position_error, rmse = compute_error_series(data, history)

    fig, axs = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(f"Harbour Surveillance Tracking Result | RMSE = {rmse:.2f} m")

    ax = axs[0, 0]

    ax.plot(gt_E[valid], gt_N[valid], label=f"Ground Truth Target {target_id}")
    ax.plot(est_E, est_N, label="EKF Track")

    radar_circle = plt.Circle(
        (0, 0),
        1000,
        fill=False,
        linestyle="--",
        alpha=0.4,
        label="Radar FOV",
    )
    ax.add_patch(radar_circle)

    camera_pos_NE = np.array([-80.0, 120.0])

    camera_circle = plt.Circle(
        (camera_pos_NE[1], camera_pos_NE[0]),
        500,
        fill=False,
        linestyle="--",
        alpha=0.4,
        label="Camera range",
    )
    ax.add_patch(camera_circle)

    ax.scatter(0, 0, marker="^", label="Radar")
    ax.scatter(camera_pos_NE[1], camera_pos_NE[0], marker="s", label="Camera")

    ax.set_xlabel("East [m]")
    ax.set_ylabel("North [m]")
    ax.set_title("2-D NED")
    ax.axis("equal")
    ax.grid(True)
    ax.legend()

    ax = axs[0, 1]

    for sensor in ["radar", "camera"]:
        t_true = []
        r_true = []
        t_false = []
        r_false = []

        for m in measurements:
            if m["sensor_id"].lower() == sensor and "range_m" in m:
                if is_false_alarm(m):
                    t_false.append(m["time"])
                    r_false.append(m["range_m"])
                else:
                    t_true.append(m["time"])
                    r_true.append(m["range_m"])

        if len(t_true) > 0:
            ax.scatter(t_true, r_true, s=8, label=f"{sensor} true")

        if len(t_false) > 0:
            ax.scatter(t_false, r_false, s=2, alpha=0.1, marker="x", label=f"{sensor} false")

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Range [m]")
    ax.set_title("Range measurements")
    ax.grid(True)
    ax.legend()

    ax = axs[1, 0]

    for sensor in ["radar", "camera"]:
        t_true = []
        b_true = []
        t_false = []
        b_false = []

        for m in measurements:
            if m["sensor_id"].lower() == sensor and "bearing_rad" in m:
                if is_false_alarm(m):
                    t_false.append(m["time"])
                    b_false.append(np.rad2deg(m["bearing_rad"]))
                else:
                    t_true.append(m["time"])
                    b_true.append(np.rad2deg(m["bearing_rad"]))

        if len(t_true) > 0:
            ax.scatter(t_true, b_true, s=8, label=f"{sensor} true")

        if len(t_false) > 0:
            ax.scatter(t_false, b_false, s=2, alpha=0.1, marker="x", label=f"{sensor} false")

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Bearing [deg]")
    ax.set_title("Bearing measurements")
    ax.grid(True)
    ax.legend()

    ax = axs[1, 1]

    ax.plot(err_times, position_error, label="Position error")
    ax.axhline(rmse, linestyle="--", label=f"RMSE = {rmse:.2f} m")

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Error [m]")
    ax.set_title("Position error over time")
    ax.grid(True)
    ax.legend()

    plt.tight_layout()
    plt.show()


# -------------------------------------------------------------------------
# Sequential tracker
# -------------------------------------------------------------------------

def run_sequential_tracker(json_path, allowed_sensors, init_sensors=("radar",), show_plots=False):
    data = load_json(json_path)

    measurements = data["measurements"]
    vessel_positions = np.asarray(data.get("vessel_positions", []), dtype=float)

    coord = CoordinateManager()

    try:
        first_meas = first_valid_measurement(measurements, init_sensors)
    except RuntimeError:
        first_meas = first_valid_measurement(measurements, allowed_sensors)

    ekf = EKF()

    x0, P0, t0 = initial_state_from_measurement(
        first_meas,
        coordinate_manager=coord,
    )

    ekf.x = x0.reshape(4, 1)
    ekf.P = P0
    ekf.t = t0

    tracker_ext = ExtendedTracker(
        ekf=ekf,
        coordinate_manager=coord,
    )

    history = tracker_ext.process_measurements_sequential(
        measurements,
        vessel_positions=vessel_positions,
        allowed_sensors=allowed_sensors,
    )

    print("Finished SEQUENTIAL tracking.")
    print("Number of updates:", len(history))
    print("Final state [N, E, vN, vE]:")
    print(ekf.x.flatten())
    print("Position RMSE [m]:", compute_rmse(data, history))

    print_nis_consistency(tracker_ext)

    if show_plots:
        plot_harbour_debug(data, history)
        plot_nis(tracker_ext)

    return tracker_ext


# -------------------------------------------------------------------------
# Joint tracker for Scenario B
# -------------------------------------------------------------------------

def find_closest_camera_measurement(camera_measurements, radar_time, tolerance_s=1.0):
    best_meas = None
    best_dt = np.inf

    for cam in camera_measurements:
        dt = abs(cam["time"] - radar_time)

        if dt < best_dt and dt <= tolerance_s:
            best_meas = cam
            best_dt = dt

    return best_meas


def run_scenario_B_joint(path, tolerance_s=1.0, show_plots=False):
    data = load_json(path)
    measurements = sorted(data["measurements"], key=lambda m: m["time"])

    coord = CoordinateManager()

    first_meas = first_valid_measurement(measurements, ("radar",))

    ekf = EKF()

    x0, P0, t0 = initial_state_from_measurement(
        first_meas,
        coordinate_manager=coord,
    )

    ekf.x = x0.reshape(4, 1)
    ekf.P = P0
    ekf.t = t0

    tracker_ext = ExtendedTracker(
        ekf=ekf,
        coordinate_manager=coord,
    )

    radar_measurements = [
        m for m in measurements
        if m["sensor_id"].lower() == "radar" and not is_false_alarm(m)
    ]

    camera_measurements = [
        m for m in measurements
        if m["sensor_id"].lower() == "camera" and not is_false_alarm(m)
    ]

    used_camera_ids = set()

    for radar_meas in radar_measurements:
        if radar_meas["time"] < t0:
            continue

        cam_meas = find_closest_camera_measurement(
            camera_measurements,
            radar_meas["time"],
            tolerance_s=tolerance_s,
        )

        if cam_meas is not None and id(cam_meas) not in used_camera_ids:
            tracker_ext.update_joint_radar_camera(radar_meas, cam_meas)
            used_camera_ids.add(id(cam_meas))
        else:
            tracker_ext.update_one(radar_meas)

    print("Finished JOINT tracking.")
    print("Number of updates:", len(tracker_ext.history))
    print("Final state [N, E, vN, vE]:")
    print(ekf.x.flatten())
    print("Position RMSE [m]:", compute_rmse(data, tracker_ext.history))

    print_nis_consistency(tracker_ext)

    if show_plots:
        plot_harbour_debug(data, tracker_ext.history)
        plot_nis(tracker_ext)

    return tracker_ext


# -------------------------------------------------------------------------
# Required validations
# -------------------------------------------------------------------------

def validate_T3_scenario_A():
    print("\n" + "=" * 70)
    print("T3 VALIDATION — SCENARIO A — RADAR ONLY")
    print("=" * 70)

    path = "harbour_sim_output/scenario_A.json"
    data = load_json(path)

    tracker = run_sequential_tracker(
        path,
        allowed_sensors=("radar",),
        init_sensors=("radar",),
        show_plots=False,
    )

    rmse = compute_rmse(data, tracker.history)
    rmse_steady = compute_rmse_in_window(data, tracker.history, 30.0, data["t_end"])
    nis = evaluate_nis_consistency(tracker)

    print("\nT3 SUMMARY")
    print(f"Overall RMSE: {rmse:.2f} m")
    print(f"Steady-state RMSE from 30 s: {rmse_steady:.2f} m")
    print(f"NIS inside bounds: {nis['percentage_inside']:.1f}%")

    print_pass_fail("Steady-state RMSE < 12 m", rmse_steady < 12.0)
    print_pass_fail("NIS >= 90%", nis["percentage_inside"] >= 90.0)

    return tracker


def validate_T4_scenario_B():
    print("\n" + "=" * 70)
    print("T4 VALIDATION — SCENARIO B — RADAR + CAMERA")
    print("=" * 70)

    path = "harbour_sim_output/scenario_B.json"
    data = load_json(path)

    print("\n--- Radar only baseline on Scenario B ---")
    tracker_radar = run_sequential_tracker(
        path,
        allowed_sensors=("radar",),
        init_sensors=("radar",),
        show_plots=False,
    )

    print("\n--- Sequential radar + camera ---")
    tracker_seq = run_sequential_tracker(
        path,
        allowed_sensors=("radar", "camera"),
        init_sensors=("radar",),
        show_plots=False,
    )

    print("\n--- Centralised / joint radar + camera ---")
    tracker_joint = run_scenario_B_joint(
        path,
        tolerance_s=1.0,
        show_plots=False,
    )

    rmse_radar = compute_rmse(data, tracker_radar.history)
    rmse_seq = compute_rmse(data, tracker_seq.history)
    rmse_joint = compute_rmse(data, tracker_joint.history)

    nis_radar = evaluate_nis_consistency(tracker_radar)
    nis_seq = evaluate_nis_consistency(tracker_seq)
    nis_joint = evaluate_nis_consistency(tracker_joint)

    improvement_seq = 100.0 * (rmse_radar - rmse_seq) / rmse_radar
    improvement_joint = 100.0 * (rmse_radar - rmse_joint) / rmse_radar

    print("\nT4 SUMMARY")
    print(f"Radar-only RMSE: {rmse_radar:.2f} m")
    print(f"Sequential RMSE: {rmse_seq:.2f} m")
    print(f"Joint RMSE: {rmse_joint:.2f} m")
    print(f"Sequential improvement: {improvement_seq:.1f}%")
    print(f"Joint improvement: {improvement_joint:.1f}%")
    print(f"Radar-only NIS: {nis_radar['percentage_inside']:.1f}%")
    print(f"Sequential NIS: {nis_seq['percentage_inside']:.1f}%")
    print(f"Joint NIS: {nis_joint['percentage_inside']:.1f}%")

    print("\nPass/fail:")
    print_pass_fail("Sequential fusion improves RMSE", rmse_seq < rmse_radar)
    print_pass_fail("Joint fusion improves RMSE", rmse_joint < rmse_radar)
    print_pass_fail("Sequential NIS >= 90%", nis_seq["percentage_inside"] >= 90.0)
    print_pass_fail("Joint NIS >= 90%", nis_joint["percentage_inside"] >= 90.0)

    print("\nBest RMSE architecture:")
    if rmse_seq < rmse_joint:
        print("Sequential update gives lower RMSE.")
    else:
        print("Centralised/joint update gives lower RMSE.")

    return tracker_radar, tracker_seq, tracker_joint


def validate_T5_scenario_C():
    print("\n" + "=" * 70)
    print("T5 VALIDATION — SCENARIO C — RADAR + CAMERA + AIS")
    print("=" * 70)

    path = "harbour_sim_output/scenario_C.json"
    data = load_json(path)

    print("\n--- Without AIS: radar + camera ---")
    tracker_no_ais = run_sequential_tracker(
        path,
        allowed_sensors=("radar", "camera"),
        init_sensors=("radar", "ais"),
        show_plots=False,
    )

    print("\n--- With AIS: radar + camera + AIS ---")
    tracker_with_ais = run_sequential_tracker(
        path,
        allowed_sensors=("radar", "camera", "ais"),
        init_sensors=("radar", "ais"),
        show_plots=False,
    )

    rmse_no_ais = compute_rmse(data, tracker_no_ais.history)
    rmse_with_ais = compute_rmse(data, tracker_with_ais.history)

    rmse_before = compute_rmse_in_window(data, tracker_with_ais.history, 0.0, 60.0)
    rmse_dropout = compute_rmse_in_window(data, tracker_with_ais.history, 60.0, 90.0)
    rmse_after = compute_rmse_in_window(data, tracker_with_ais.history, 90.0, data["t_end"])

    updates_dropout = compute_update_count_in_window(tracker_with_ais.history, 60.0, 90.0)

    nis_no_ais = evaluate_nis_consistency(tracker_no_ais)
    nis_with_ais = evaluate_nis_consistency(tracker_with_ais)

    improvement = 100.0 * (rmse_no_ais - rmse_with_ais) / rmse_no_ais

    print("\nT5 SUMMARY")
    print(f"Without AIS RMSE: {rmse_no_ais:.2f} m")
    print(f"With AIS RMSE: {rmse_with_ais:.2f} m")
    print(f"AIS improvement: {improvement:.1f}%")
    print(f"RMSE before dropout 0-60 s: {rmse_before:.2f} m")
    print(f"RMSE during dropout 60-90 s: {rmse_dropout:.2f} m")
    print(f"RMSE after dropout 90-end s: {rmse_after:.2f} m")
    print(f"Updates during dropout: {updates_dropout}")
    print(f"NIS without AIS: {nis_no_ais['percentage_inside']:.1f}%")
    print(f"NIS with AIS: {nis_with_ais['percentage_inside']:.1f}%")

    print("\nPass/fail:")
    print_pass_fail("AIS improves RMSE", rmse_with_ais < rmse_no_ais)
    print_pass_fail("Track survives AIS dropout", updates_dropout > 0 and not np.isnan(rmse_dropout))
    print_pass_fail("Smooth reacquisition after dropout", not np.isnan(rmse_after))
    print_pass_fail("NIS with AIS >= 90%", nis_with_ais["percentage_inside"] >= 90.0)

    return tracker_no_ais, tracker_with_ais


def validate_all_required_tasks():
    tracker_A = validate_T3_scenario_A()
    tracker_B_radar, tracker_B_seq, tracker_B_joint = validate_T4_scenario_B()
    tracker_C_no_ais, tracker_C_with_ais = validate_T5_scenario_C()

    print("\n" + "=" * 70)
    print("ALL REQUIRED VALIDATIONS FINISHED")
    print("=" * 70)

    return {
        "T3_A": tracker_A,
        "T4_B_radar": tracker_B_radar,
        "T4_B_sequential": tracker_B_seq,
        "T4_B_joint": tracker_B_joint,
        "T5_C_no_ais": tracker_C_no_ais,
        "T5_C_with_ais": tracker_C_with_ais,
    }


if __name__ == "__main__":
    validate_all_required_tasks()