import json
import numpy as np
import matplotlib.pyplot as plt

from ekf import EKF
from coordinate_manager import CoordinateManager
from extended_tracker import ExtendedTracker, initial_state_from_measurement


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def first_valid_measurement(measurements, allowed_sensors):
    for m in sorted(measurements, key=lambda x: x["time"]):
        sid = m["sensor_id"].lower()

        if sid not in allowed_sensors:
            continue

        if int(m.get("true_target_id", m.get("target_id", 0))) == -1:
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

    gt_time_valid = gt_time[valid]
    gt_N_valid = gt_N[valid]
    gt_E_valid = gt_E[valid]

    gt_N_interp = np.interp(times, gt_time_valid, gt_N_valid)
    gt_E_interp = np.interp(times, gt_time_valid, gt_E_valid)

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
    count = 0

    for h in history:
        if t_start <= h["time"] <= t_end:
            count += 1

    return count


def print_pass_fail(name, passed, value_text=""):
    symbol = "PASS" if passed else "FAIL"
    print(f"{symbol}: {name} {value_text}")


# -------------------------------------------------------------------------
# NIS FUNCTIONS
# -------------------------------------------------------------------------

def evaluate_nis_consistency(tracker_ext, ignore_first=0):
    """
    Evaluates NIS consistency.

    Since radar, camera, and AIS converted measurements are:
        z = [range, bearing]

    the measurement dimension is 2.

    95% chi-square bounds for 2 DOF:
        lower = 0.103
        upper = 5.991

    ignore_first:
        Can be used to ignore initial transient updates.
        For official reporting, keep it at 0 unless you clearly state otherwise.
    """

    if len(tracker_ext.nis_history) == 0:
        return {
            "available": False,
            "percentage_inside": np.nan,
            "mean_nis": np.nan,
            "lower": 0.103,
            "upper": 5.991,
            "n": 0,
            "by_sensor": {},
        }

    nis_items = tracker_ext.nis_history[ignore_first:]

    if len(nis_items) == 0:
        return {
            "available": False,
            "percentage_inside": np.nan,
            "mean_nis": np.nan,
            "lower": 0.103,
            "upper": 5.991,
            "n": 0,
            "by_sensor": {},
        }

    lower = 0.103
    upper = 5.991

    nis_values = np.array([item["nis"] for item in nis_items], dtype=float)
    inside = (nis_values >= lower) & (nis_values <= upper)

    by_sensor = {}

    for sensor in sorted(set(item["sensor"] for item in nis_items)):
        vals = np.array(
            [item["nis"] for item in nis_items if item["sensor"] == sensor],
            dtype=float,
        )

        inside_sensor = (vals >= lower) & (vals <= upper)

        by_sensor[sensor] = {
            "n": len(vals),
            "mean_nis": float(np.mean(vals)),
            "percentage_inside": 100.0 * np.mean(inside_sensor),
        }

    return {
        "available": True,
        "percentage_inside": 100.0 * np.mean(inside),
        "mean_nis": float(np.mean(nis_values)),
        "lower": lower,
        "upper": upper,
        "n": len(nis_values),
        "by_sensor": by_sensor,
    }


def print_nis_consistency(tracker_ext, ignore_first=0):
    result = evaluate_nis_consistency(tracker_ext, ignore_first=ignore_first)

    print("\nNIS CONSISTENCY TEST")
    print("-" * 40)

    if not result["available"]:
        print("No NIS values available.")
        return result

    print(f"Number of NIS samples: {result['n']}")
    print(f"Mean NIS: {result['mean_nis']:.3f}")
    print(f"95% chi-square bounds: [{result['lower']:.3f}, {result['upper']:.3f}]")
    print(f"Percentage inside bounds: {result['percentage_inside']:.1f}%")

    print_pass_fail(
        "> 90% NIS inside 95% bounds",
        result["percentage_inside"] >= 90.0,
        f"({result['percentage_inside']:.1f}%)",
    )

    print("\nPer-sensor NIS:")
    for sensor, stats in result["by_sensor"].items():
        print(
            f"- {sensor}: "
            f"{stats['percentage_inside']:.1f}% inside, "
            f"mean NIS = {stats['mean_nis']:.2f}, "
            f"n = {stats['n']}"
        )

    return result


def plot_nis(tracker_ext):
    if len(tracker_ext.nis_history) == 0:
        print("No NIS values to plot.")
        return

    nis_times = np.array([item["time"] for item in tracker_ext.nis_history])
    nis_values = np.array([item["nis"] for item in tracker_ext.nis_history])
    nis_sensors = [item["sensor"] for item in tracker_ext.nis_history]

    lower = 0.103
    upper = 5.991

    plt.figure(figsize=(9, 4))

    for sensor in sorted(set(nis_sensors)):
        idx = [i for i, s in enumerate(nis_sensors) if s == sensor]
        plt.scatter(nis_times[idx], nis_values[idx], s=15, label=sensor)

    plt.axhline(lower, linestyle="--", label="95% lower bound")
    plt.axhline(upper, linestyle="--", label="95% upper bound")

    plt.xlabel("Time [s]")
    plt.ylabel("NIS")
    plt.title("NIS consistency test")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def print_success_criteria(data, history, allowed_sensors, tracker_ext=None):
    scenario = str(data.get("scenario_name", "")).upper()
    rmse = compute_rmse(data, history)

    nis_result = None
    if tracker_ext is not None:
        nis_result = evaluate_nis_consistency(tracker_ext)

    print("\n" + "=" * 60)
    print(f"SCENARIO {scenario} SUCCESS CRITERIA")
    print("=" * 60)

    if scenario == "A":
        print("Expected criteria:")
        print("- Radar-only single target")
        print("- Track confirmed within 5 scans")
        print("- Steady-state RMSE < 12 m")
        print("- > 90% NIS inside 95% chi-square bounds")

        rmse_steady = compute_rmse_in_window(data, history, 30.0, data["t_end"])

        print("\nMeasured:")
        print(f"- Overall RMSE: {rmse:.2f} m")
        print(f"- Steady-state RMSE from 30 s: {rmse_steady:.2f} m")
        print(f"- Number of EKF updates: {len(history)}")

        if nis_result is not None and nis_result["available"]:
            print(f"- NIS inside 95% bounds: {nis_result['percentage_inside']:.1f}%")
            print(f"- Mean NIS: {nis_result['mean_nis']:.2f}")

        print("\nPass/fail:")
        print_pass_fail(
            "Steady-state RMSE < 12 m",
            rmse_steady < 12.0,
            f"({rmse_steady:.2f} m)",
        )

        if nis_result is not None and nis_result["available"]:
            print_pass_fail(
                "> 90% NIS inside 95% bounds",
                nis_result["percentage_inside"] >= 90.0,
                f"({nis_result['percentage_inside']:.1f}%)",
            )

        print("INFO: Track confirmation needs T7 track management.")

    elif scenario == "B":
        print("Expected criteria:")
        print("- Single target with radar + stereo camera")
        print("- Target crosses camera FOV around t = 20-80 s")
        print("- Quantitative RMSE improvement: camera fusion vs radar-only")
        print("- NIS consistent for sequential and centralised architectures")

        rmse_fov = compute_rmse_in_window(data, history, 20.0, 80.0)

        print("\nMeasured:")
        print(f"- Overall RMSE: {rmse:.2f} m")
        print(f"- RMSE during camera-FOV window 20-80 s: {rmse_fov:.2f} m")
        print(f"- Sensors used: {allowed_sensors}")

        if nis_result is not None and nis_result["available"]:
            print(f"- NIS inside 95% bounds: {nis_result['percentage_inside']:.1f}%")
            print(f"- Mean NIS: {nis_result['mean_nis']:.2f}")

        print("\nPass/fail:")

        if nis_result is not None and nis_result["available"]:
            print_pass_fail(
                "NIS consistency for this run",
                nis_result["percentage_inside"] >= 90.0,
                f"({nis_result['percentage_inside']:.1f}%)",
            )

        print("INFO: Full Scenario B validation requires radar-only vs radar+camera comparison.")
        print("INFO: Centralised/joint update still needs to be run separately if required.")

    elif scenario == "C":
        print("Expected criteria:")
        print("- AIS-equipped target")
        print("- All sensors active")
        print("- AIS dropout from t = 60-90 s")
        print("- Track survives dropout")
        print("- RMSE lower with AIS during available windows")
        print("- Smooth reacquisition after dropout")

        rmse_before_dropout = compute_rmse_in_window(data, history, 0.0, 60.0)
        rmse_dropout = compute_rmse_in_window(data, history, 60.0, 90.0)
        rmse_after_dropout = compute_rmse_in_window(data, history, 90.0, data["t_end"])
        updates_dropout = compute_update_count_in_window(history, 60.0, 90.0)

        print("\nMeasured:")
        print(f"- Overall RMSE: {rmse:.2f} m")
        print(f"- RMSE before AIS dropout 0-60 s: {rmse_before_dropout:.2f} m")
        print(f"- RMSE during AIS dropout 60-90 s: {rmse_dropout:.2f} m")
        print(f"- RMSE after AIS dropout 90-end s: {rmse_after_dropout:.2f} m")
        print(f"- Updates during dropout: {updates_dropout}")

        if nis_result is not None and nis_result["available"]:
            print(f"- NIS inside 95% bounds: {nis_result['percentage_inside']:.1f}%")
            print(f"- Mean NIS: {nis_result['mean_nis']:.2f}")

        print("\nPass/fail:")
        print_pass_fail(
            "Track survives AIS dropout",
            updates_dropout > 0 and not np.isnan(rmse_dropout),
            f"(updates during dropout = {updates_dropout})",
        )

        print_pass_fail(
            "Reacquisition after dropout",
            not np.isnan(rmse_after_dropout),
            f"(after-dropout RMSE = {rmse_after_dropout:.2f} m)",
        )

        if nis_result is not None and nis_result["available"]:
            print_pass_fail(
                "> 90% NIS inside 95% bounds",
                nis_result["percentage_inside"] >= 90.0,
                f"({nis_result['percentage_inside']:.1f}%)",
            )

        print("INFO: To prove AIS improvement, run Scenario C twice:")
        print("      1) allowed_sensors=('radar','camera')")
        print("      2) allowed_sensors=('radar','camera','ais')")

    elif scenario == "D":
        print("Expected criteria:")
        print("- 4 crossing targets")
        print("- All 4 tracks maintained")
        print("- No identity swap")
        print("- MOTP < 15 m")
        print("- CE < 0.5")
        print("\nINFO: Requires T6/T7 multi-target tracking.")

    elif scenario == "E":
        print("Expected criteria:")
        print("- 6 targets, mixed AIS and non-AIS")
        print("- All 6 targets tracked")
        print("- MOTP < 20 m")
        print("- CE < 1.0")
        print("\nINFO: Requires complete T6/T7 pipeline.")

    else:
        print("Unknown scenario.")

    print("=" * 60 + "\n")


def compare_scenario_B_fusion():
    """
    Scenario B validation.

    Required by the project:
        - Compare radar-only against radar + camera on Scenario B.
        - Check RMSE improvement.
        - Check NIS consistency for the sequential architecture.

    Centralised/joint update can be added later.
    """

    path = "harbour_sim_output/scenario_B.json"
    data = load_json(path)

    print("\n" + "=" * 70)
    print("SCENARIO B FUSION COMPARISON")
    print("=" * 70)

    print("\n--- Scenario B: radar only ---")
    tracker_radar = main(
        path,
        allowed_sensors=("radar",),
        show_plots=False,
    )

    rmse_radar = compute_rmse(data, tracker_radar.history)
    nis_radar = evaluate_nis_consistency(tracker_radar)

    print("\n--- Scenario B: radar + camera sequential ---")
    tracker_seq = main(
        path,
        allowed_sensors=("radar", "camera"),
        show_plots=False,
    )

    rmse_seq = compute_rmse(data, tracker_seq.history)
    nis_seq = evaluate_nis_consistency(tracker_seq)

    improvement = 100.0 * (rmse_radar - rmse_seq) / rmse_radar

    print("\n" + "=" * 70)
    print("SCENARIO B SUMMARY")
    print("=" * 70)

    print(f"Radar-only RMSE                    : {rmse_radar:.2f} m")
    print(f"Radar + camera sequential RMSE     : {rmse_seq:.2f} m")
    print(f"RMSE improvement                   : {improvement:.1f}%")

    print(f"Radar-only NIS inside 95%          : {nis_radar['percentage_inside']:.1f}%")
    print(f"Sequential NIS inside 95%          : {nis_seq['percentage_inside']:.1f}%")

    print("\nPass/fail:")
    print_pass_fail(
        "Camera fusion improves RMSE",
        rmse_seq < rmse_radar,
        f"({rmse_radar:.2f} m -> {rmse_seq:.2f} m)",
    )

    print_pass_fail(
        "Sequential NIS >= 90%",
        nis_seq["percentage_inside"] >= 90.0,
        f"({nis_seq['percentage_inside']:.1f}%)",
    )

    print("INFO: Add centralised/joint update result when implemented.")
    print("=" * 70)

    plot_harbour_debug(data, tracker_seq.history)
    plot_nis(tracker_seq)


def compare_scenarios_A_B_C():
    print("\n" + "=" * 70)
    print("SCENARIO A/B/C COMPARISON")
    print("=" * 70)

    results = []

    experiments = [
        ("Scenario A", "harbour_sim_output/scenario_A.json", ("radar",)),
        ("Scenario B", "harbour_sim_output/scenario_B.json", ("radar", "camera")),
        ("Scenario C", "harbour_sim_output/scenario_C.json", ("radar", "camera", "ais")),
    ]

    for name, path, sensors in experiments:
        print("\n" + "-" * 70)
        print(f"Running {name} with sensors {sensors}")
        print("-" * 70)

        tracker = main(path, allowed_sensors=sensors, show_plots=False)

        data = load_json(path)
        rmse = compute_rmse(data, tracker.history)
        nis = evaluate_nis_consistency(tracker)

        results.append(
            {
                "scenario": name,
                "sensors": sensors,
                "rmse": rmse,
                "updates": len(tracker.history),
                "nis_inside": nis["percentage_inside"],
            }
        )

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for r in results:
        print(
            f"{r['scenario']:<12} | "
            f"Sensors: {str(r['sensors']):<30} | "
            f"RMSE: {r['rmse']:.2f} m | "
            f"NIS inside: {r['nis_inside']:.1f}% | "
            f"Updates: {r['updates']}"
        )

    print("=" * 70)


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

    legend_added = set()

    for m in measurements:
        sensor = m["sensor_id"].lower()

        is_false = (
            m.get("is_false_alarm", False)
            or int(m.get("true_target_id", m.get("target_id", 0))) == -1
        )

        if sensor in ["radar", "camera"] and "range_m" in m and "bearing_rad" in m:
            r = m["range_m"]
            theta = m["bearing_rad"]

            if sensor == "radar":
                x = r * np.sin(theta)
                y = r * np.cos(theta)

            elif sensor == "camera":
                cam_n, cam_e = -80.0, 120.0
                x = cam_e + r * np.sin(theta)
                y = cam_n + r * np.cos(theta)

            color = "blue" if sensor == "radar" else "red"
            marker = "x" if is_false else "o"
            alpha = 0.1 if is_false else 0.7
            size = 3 if is_false else 20
            label = f"{sensor} {'false' if is_false else 'true'}"

            if label not in legend_added:
                ax.scatter(x, y, marker=marker, alpha=alpha, color=color, s=size, label=label)
                legend_added.add(label)
            else:
                ax.scatter(x, y, marker=marker, alpha=alpha, color=color, s=size)

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
                is_false = (
                    m.get("is_false_alarm", False)
                    or int(m.get("true_target_id", m.get("target_id", 0))) == -1
                )

                if is_false:
                    t_false.append(m["time"])
                    r_false.append(m["range_m"])
                else:
                    t_true.append(m["time"])
                    r_true.append(m["range_m"])

        if len(t_true) > 0:
            ax.scatter(t_true, r_true, s=8, label=f"{sensor} true")

        if len(t_false) > 0:
            ax.scatter(t_false, r_false, s=2, label=f"{sensor} false", alpha=0.1, marker="x")

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
                is_false = (
                    m.get("is_false_alarm", False)
                    or int(m.get("true_target_id", m.get("target_id", 0))) == -1
                )

                if is_false:
                    t_false.append(m["time"])
                    b_false.append(np.rad2deg(m["bearing_rad"]))
                else:
                    t_true.append(m["time"])
                    b_true.append(np.rad2deg(m["bearing_rad"]))

        if len(t_true) > 0:
            ax.scatter(t_true, b_true, s=8, label=f"{sensor} true")

        if len(t_false) > 0:
            ax.scatter(t_false, b_false, s=2, label=f"{sensor} false", alpha=0.1, marker="x")

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


def main(json_path, allowed_sensors=("radar", "camera", "ais"), show_plots=True):
    data = load_json(json_path)

    measurements = data["measurements"]
    vessel_positions = np.asarray(data.get("vessel_positions", []), dtype=float)

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

    history = tracker_ext.process_measurements_sequential(
        measurements,
        vessel_positions=vessel_positions,
        allowed_sensors=allowed_sensors,
    )

    print("Finished tracking.")
    print("Number of updates:", len(history))
    print("Final state [N, E, vN, vE]:")
    print(ekf.x.flatten())

    rmse = compute_rmse(data, history)
    print("Position RMSE [m]:", rmse)

    print_nis_consistency(tracker_ext)
    print_success_criteria(data, history, allowed_sensors, tracker_ext)

    if show_plots:
        plot_harbour_debug(data, history)
        plot_nis(tracker_ext)

    return tracker_ext


if __name__ == "__main__":
    compare_scenario_B_fusion()

    # Optional:
    #compare_scenarios_A_B_C()