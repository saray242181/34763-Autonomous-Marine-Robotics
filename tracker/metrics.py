import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment


MOTP_REJECT_THRESHOLD = 50.0  # metres — reject matched pairs beyond this


def evaluate_tracker_performance(
    ground_truth: dict,
    tracker_history: list,
) -> tuple[float, float]:
    """
    Computes and plots MOTP and CE metrics over time.

    Args:
        ground_truth: Dictionary of true target trajectories.
            Keys: target IDs (str or int).
            Values: (T×5) arrays of [time, N, E, vN, vE]; NaN rows indicate inactive target.
        tracker_history: List of dicts per scan.
            Format: [{"time": float, "tracks": [{"id": int, "pos": [N, E], "state": str}, ...]}, ...]

    Returns:
        mean_motp: Scalar average MOTP [m] across all valid timesteps.
        mean_ce: Scalar average CE across all timesteps.
    """
    timestamps = []
    motp_series = []
    ce_series = []

    # Pre-build per-target valid (non-NaN) time/N/E arrays for fast interpolation
    gt_valid = {}
    for tid, arr in ground_truth.items():
        arr = np.asarray(arr, dtype=float)
        valid_mask = ~np.isnan(arr[:, 1])
        if valid_mask.any():
            gt_valid[tid] = {
                "t": arr[valid_mask, 0],
                "N": arr[valid_mask, 1],
                "E": arr[valid_mask, 2],
                "t_min": arr[valid_mask, 0].min(),
                "t_max": arr[valid_mask, 0].max(),
            }

    for scan in tracker_history:
        t_k = float(scan["time"])
        timestamps.append(t_k)

        # Step 3A: interpolate active GT targets at t_k
        active_gt = []
        for tid, vd in gt_valid.items():
            if vd["t_min"] <= t_k <= vd["t_max"]:
                N_interp = float(np.interp(t_k, vd["t"], vd["N"]))
                E_interp = float(np.interp(t_k, vd["t"], vd["E"]))
                active_gt.append(np.array([N_interp, E_interp]))

        # Step 3B: Cardinality Error
        # Coasting tracks have met M-of-N and represent real targets temporarily
        # missing detections; they count toward CE and MOTP like confirmed tracks.
        confirmed_tracks = [
            tr for tr in scan.get("tracks", [])
            if tr.get("state") in ("confirmed", "coasting")
        ]
        n_true = len(active_gt)
        n_tracks = len(confirmed_tracks)
        ce_series.append(abs(n_tracks - n_true))

        # Step 3C: MOTP
        if n_true == 0 or n_tracks == 0:
            motp_series.append(np.nan)
            continue

        track_pos = np.array([tr["pos"] for tr in confirmed_tracks], dtype=float)  # (n_tracks, 2)
        gt_pos = np.array(active_gt, dtype=float)                                  # (n_true, 2)

        # Euclidean distance cost matrix: rows = tracks, cols = GT targets
        diff = track_pos[:, np.newaxis, :] - gt_pos[np.newaxis, :, :]  # (n_tracks, n_true, 2)
        cost = np.linalg.norm(diff, axis=2)                             # (n_tracks, n_true)

        try:
            row_ind, col_ind = linear_sum_assignment(cost)
        except ValueError:
            motp_series.append(np.nan)
            continue

        valid_dists = [
            cost[r, c]
            for r, c in zip(row_ind, col_ind)
            if cost[r, c] <= MOTP_REJECT_THRESHOLD
        ]

        motp_series.append(float(np.mean(valid_dists)) if valid_dists else np.nan)

    # Step 3D: Aggregation
    timestamps = np.array(timestamps)
    motp_arr = np.array(motp_series, dtype=float)
    ce_arr = np.array(ce_series, dtype=float)

    mean_motp = float(np.nanmean(motp_arr)) if not np.all(np.isnan(motp_arr)) else float("nan")
    mean_ce = float(np.nanmean(ce_arr)) if ce_arr.size > 0 else float("nan")

    # Plotting
    fig, (ax_motp, ax_ce) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    ax_motp.plot(timestamps, motp_arr, color="steelblue", linewidth=1.2, label="MOTP")
    ax_motp.axhline(mean_motp, color="steelblue", linestyle="--", linewidth=1.0,
                    label=f"Mean = {mean_motp:.2f} m")
    ax_motp.set_ylabel("Localization Error [m]")
    ax_motp.set_title(f"MOTP (Mean: {mean_motp:.2f} m)")
    ax_motp.grid(True)
    ax_motp.legend()

    ax_ce.plot(timestamps, ce_arr, color="darkorange", linewidth=1.2, label="CE")
    ax_ce.axhline(mean_ce, color="darkorange", linestyle="--", linewidth=1.0,
                  label=f"Mean = {mean_ce:.2f}")
    ax_ce.set_xlabel("Time [s]")
    ax_ce.set_ylabel("Absolute Track Count Error")
    ax_ce.set_title(f"Cardinality Error (Mean: {mean_ce:.2f})")
    ax_ce.grid(True)
    ax_ce.legend()

    plt.tight_layout()
    plt.show()

    return mean_motp, mean_ce
