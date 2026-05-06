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

        if m.get("true_target_id", 0) == -1:
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

    gt_N = truth[:, 0]
    gt_E = truth[:, 1]

    valid = ~np.isnan(gt_N)

    return gt_N, gt_E, valid, target_id


def compute_rmse(data, history):
    _, est_N, est_E = extract_estimates(history)

    gt_N, gt_E, valid, _ = get_first_ground_truth(data)

    n = min(len(est_N), np.sum(valid))

    if n == 0:
        return np.nan

    rmse = np.sqrt(
        np.mean(
            (est_N[:n] - gt_N[valid][:n]) ** 2
            + (est_E[:n] - gt_E[valid][:n]) ** 2
        )
    )

    return rmse


def plot_harbour_debug(data, history):
    measurements = data["measurements"]

    times, est_N, est_E = extract_estimates(history)
    gt_N, gt_E, valid, target_id = get_first_ground_truth(data)

    fig, axs = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Harbour Surveillance Tracking Result")

    # --------------------------------------------------
    # 1. 2-D NED scene
    # --------------------------------------------------
    ax = axs[0, 0]

    ax.plot(gt_E[valid], gt_N[valid], label=f"Ground Truth Target {target_id}")
    ax.plot(est_E, est_N, label="EKF Track")

    # Radar FOV
    radar_circle = plt.Circle(
        (0, 0),          # x = East, y = North
        1000,
        fill=False,
        linestyle="--",
        alpha=0.4,
        label="Radar FOV",
        color="green",
    )
    ax.add_patch(radar_circle)

    # Camera approximate FOV/range circle
    camera_pos_NE = np.array([-80.0, 120.0])  # [N, E]
    camera_circle = plt.Circle(
        (camera_pos_NE[1], camera_pos_NE[0]),  # x = E, y = N
        500,
        fill=False,
        linestyle="--",
        alpha=0.4,
        label="Camera range",
        color="red",
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

    # --------------------------------------------------
    # 2. Range measurements
    # --------------------------------------------------
    ax = axs[0, 1]

    for sensor in ["radar", "camera"]:
        t = []
        r = []

        for m in measurements:
            if m["sensor_id"].lower() == sensor and "range_m" in m:
                t.append(m["time"])
                r.append(m["range_m"])

        if len(t) > 0:
            ax.scatter(t, r, s=8, label=sensor)

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Range [m]")
    ax.set_title("Range measurements")
    ax.grid(True)
    ax.legend()

    # --------------------------------------------------
    # 3. Bearing measurements
    # --------------------------------------------------
    ax = axs[1, 0]

    for sensor in ["radar", "camera"]:
        t = []
        b = []

        for m in measurements:
            if m["sensor_id"].lower() == sensor and "bearing_rad" in m:
                t.append(m["time"])
                b.append(np.rad2deg(m["bearing_rad"]))

        if len(t) > 0:
            ax.scatter(t, b, s=8, label=sensor)

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Bearing [deg]")
    ax.set_title("Bearing measurements")
    ax.grid(True)
    ax.legend()

    # --------------------------------------------------
    # 4. EKF estimated position over time
    # --------------------------------------------------
    ax = axs[1, 1]

    ax.plot(times, est_N, label="Estimated North")
    ax.plot(times, est_E, label="Estimated East")

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Position [m]")
    ax.set_title("EKF estimated states")
    ax.grid(True)
    ax.legend()

    plt.tight_layout()
    plt.show()


def main(json_path, allowed_sensors=("radar", "camera", "ais")):
    data = load_json(json_path)

    measurements = data["measurements"]
    vessel_positions = np.asarray(data.get("vessel_positions", []), dtype=float)

    coord = CoordinateManager()

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

    print("Finished tracking.")
    print("Number of updates:", len(history))
    print("Final state [N, E, vN, vE]:")
    print(ekf.x.flatten())

    rmse = compute_rmse(data, history)
    print("Position RMSE [m]:", rmse)

    plot_harbour_debug(data, history)

    return tracker_ext


if __name__ == "__main__":
    # Scenario A: radar only
    tracker = main(
        "harbour_sim_output/scenario_A.json",
        allowed_sensors=("radar",),
    )

    # Scenario B: radar + camera
    #
    # tracker = main(
    #     "harbour_sim_output/scenario_B.json",
    #     allowed_sensors=("radar", "camera"),
    # )

    # Scenario C: radar + camera + AIS
    #
    # tracker = main(
    #     "harbour_sim_output/scenario_C.json",
    #     allowed_sensors=("radar", "camera", "ais"),
    # )