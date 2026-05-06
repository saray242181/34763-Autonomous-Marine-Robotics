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


def plot_results(data, history):
    times, est_N, est_E = extract_estimates(history)

    gt = data["ground_truth"]
    target_id = list(gt.keys())[0]
    truth = np.asarray(gt[target_id], dtype=float)

    gt_N = truth[:, 0]
    gt_E = truth[:, 1]

    valid = ~np.isnan(gt_N)

    plt.figure(figsize=(8, 6))
    plt.plot(gt_E[valid], gt_N[valid], label="Ground Truth")
    plt.plot(est_E, est_N, label="EKF Track")

    plt.xlabel("East [m]")
    plt.ylabel("North [m]")
    plt.title("Ground Truth vs EKF")
    plt.axis("equal")
    plt.grid()
    plt.legend()
    plt.show()

    n = min(len(est_N), np.sum(valid))

    rmse = np.sqrt(
        np.mean(
            (est_N[:n] - gt_N[valid][:n]) ** 2
            + (est_E[:n] - gt_E[valid][:n]) ** 2
        )
    )

    print("Position RMSE [m]:", rmse)


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

    plot_results(data, history)

    return tracker_ext


if __name__ == "__main__":
    tracker = main(
        "harbour_sim_output/scenario_B.json",
        allowed_sensors=("radar", "camera"),
    )

    # For Scenario C with AIS, use:
    #
    # tracker = main(
    #     "harbour_sim_output/scenario_C.json",
    #     allowed_sensors=("radar", "camera", "ais"),
    # )