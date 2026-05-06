"""
gnn_tracker.py  —  Task 6 (GNN) + Task 7 (Track Lifecycle Management)

Builds on the existing stack without modifying it:
    ekf.py               — EKF.predict_to(), EKF.update(z, z_pred, H, R)
    coordinate_manager.py — CoordinateManager.h_range_bearing / H_range_bearing
                            / R / measurement_to_z / ned_from_range_bearing
    extended_tracker.py  — untouched; ExtendedTracker still valid for T3–T5

Public surface
--------------
Track                          — EKF wrapper with full T7 lifecycle state
compute_mahalanobis_distance() — squared Mahalanobis d² with bearing wrap
associate_gnn()                — Hungarian GNN with 99 % chi²(2) gate
MultiTargetTracker             — predict → associate → update → lifecycle loop

Guardrails (NOT implemented here)
-----------------------------------
- MOTP and Cardinality Error metric calculations (separate evaluation script)
- Any modification to GNN association logic from T6
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Set, Tuple, Any

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import chi2

from ekf import EKF


# ---------------------------------------------------------------------------
# Shared utility
# ---------------------------------------------------------------------------

def _wrap_angle(a: float | np.ndarray) -> float | np.ndarray:
    """Wrap angle(s) to (-pi, pi]."""
    return (a + np.pi) % (2.0 * np.pi) - np.pi


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

# --- T6: GNN gate ---
_GATE_PROB      = 0.99
_DOF_MEAS       = 2
GATE_THRESHOLD: float = float(chi2.ppf(_GATE_PROB, df=_DOF_MEAS))   # ≈ 9.21

# --- T7: Track lifecycle ---
M_CONFIRM: int   = 3     # minimum hits inside window to confirm a tentative track
N_WINDOW:  int   = 5     # sliding history window length (scans)
K_DEL:     int   = 5     # consecutive misses before a track is deleted
MAX_SPEED: float = 15.0  # m/s — hard vessel speed cap for 2-point initiation

# --- T7: Track merging ---
_DOF_STATE = 4
MERGE_THRESHOLD: float = float(chi2.ppf(0.99, df=_DOF_STATE))       # ≈ 13.28


# ---------------------------------------------------------------------------
# Track  —  EKF wrapper with full lifecycle state
# ---------------------------------------------------------------------------

class Track:
    """
    Wraps a single EKF instance with T7 lifecycle state and sliding history.

    States
    ------
    "tentative" : newly spawned, not yet confirmed  (initial state)
    "confirmed" : seen >= M_CONFIRM times in last N_WINDOW scans
    "coasting"  : confirmed track currently missing; predict-only, covariance grows
    "deleted"   : >= K_DEL consecutive misses; will be purged at end of scan

    Transitions
    -----------
    tentative  → confirmed  : sum(history[-N_WINDOW:]) >= M_CONFIRM
    tentative  → deleted    : missed_detections >= K_DEL
    confirmed  → coasting   : first missed detection
    coasting   → confirmed  : detection matched again
    coasting   → deleted    : missed_detections >= K_DEL
    confirmed/coasting → deleted : merged away (smaller hit_streak loses)

    History
    -------
    A deque-like list of booleans (True = hit, False = miss) capped at
    N_WINDOW entries.  Only the last N_WINDOW entries determine confirmation.
    """

    def __init__(self, track_id: int, ekf: EKF) -> None:
        self.track_id: int             = track_id
        self.ekf                       = ekf
        self.missed_detections: int    = 0
        self.hit_streak: int           = 1
        self.state: str                = "tentative"
        self.history: List[bool]       = []

    # ------------------------------------------------------------------
    # Core interface (called by MultiTargetTracker after each scan)
    # ------------------------------------------------------------------

    def predict(self, time_s: float) -> None:
        """Advance EKF prediction to time_s."""
        self.ekf.predict_to(time_s)

    def mark_matched(self) -> None:
        """Record a successful detection association for this scan."""
        self.history.append(True)
        if len(self.history) > N_WINDOW:
            self.history = self.history[-N_WINDOW:]

        self.missed_detections = 0
        self.hit_streak += 1

        if self.state == "tentative" and sum(self.history) >= M_CONFIRM:
            self.state = "confirmed"
        elif self.state == "coasting":
            self.state = "confirmed"

    def mark_missed(self) -> None:
        """Record a missed detection for this scan."""
        self.history.append(False)
        if len(self.history) > N_WINDOW:
            self.history = self.history[-N_WINDOW:]

        self.missed_detections += 1

        if self.missed_detections >= K_DEL:
            self.state = "deleted"
        elif self.state == "confirmed":
            self.state = "coasting"

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def pos_ned(self) -> np.ndarray:
        """Current [N, E] estimate as a flat array."""
        return np.asarray(self.ekf.x, dtype=float).flatten()[:2]

    def __repr__(self) -> str:
        p = self.pos_ned
        return (
            f"Track(id={self.track_id}, state={self.state}, "
            f"N={p[0]:.1f} E={p[1]:.1f}, "
            f"hits={self.hit_streak}, missed={self.missed_detections}, "
            f"history={''.join('H' if h else 'M' for h in self.history)})"
        )


# ---------------------------------------------------------------------------
# Mahalanobis distance  (T6 — unchanged)
# ---------------------------------------------------------------------------

def compute_mahalanobis_distance(
    track: Track,
    z_meas: np.ndarray,
    sensor_pos: np.ndarray,
    sensor_id: str,
    coord_manager: Any,
    vessel_pos: Optional[np.ndarray] = None,
) -> float:
    """
    Squared Mahalanobis distance between a track's prediction and one detection.

        d² = y^T  S^{-1}  y
        y  = z_meas − h(x_pred)        innovation
        S  = H P⁻ H^T + R              innovation covariance

    Bearing element y[1] is wrapped to (−π, π] before computing d².
    Returns np.inf if S is singular.
    """
    x = track.ekf.x
    P = track.ekf.P

    sensor_pos = np.asarray(sensor_pos, dtype=float).reshape(2)
    z_meas     = np.asarray(z_meas,    dtype=float).reshape(-1)

    z_pred = coord_manager.h_range_bearing(x, sensor_pos)
    H      = coord_manager.H_range_bearing(x, sensor_pos)
    R      = coord_manager.R(sensor_id, x_pred=x, vessel_pos=vessel_pos)

    S    = H @ P @ H.T + R
    y    = z_meas - z_pred
    y[1] = float(_wrap_angle(y[1]))

    try:
        d2 = float(y @ np.linalg.inv(S) @ y)
    except np.linalg.LinAlgError:
        return np.inf

    return d2


# ---------------------------------------------------------------------------
# GNN association  (T6 — unchanged)
# ---------------------------------------------------------------------------

def associate_gnn(
    active_tracks: List[Track],
    detections_z: List[np.ndarray],
    sensor_positions: List[np.ndarray],
    sensor_id: str,
    coord_manager: Any,
    vessel_pos: Optional[np.ndarray] = None,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """
    Global Nearest Neighbour association via the Hungarian algorithm.

    Gate: chi²(2) at P_G = 0.99  →  ≈ 9.21

    Returns
    -------
    matched_pairs        : list of (track_index, detection_index)
    unmatched_tracks     : list of track_index integers
    unmatched_detections : list of detection_index integers
    """
    n_tracks = len(active_tracks)
    n_dets   = len(detections_z)

    if n_tracks == 0:
        return [], [], list(range(n_dets))
    if n_dets == 0:
        return [], list(range(n_tracks)), []

    cost = np.full((n_tracks, n_dets), np.inf, dtype=float)

    for i, track in enumerate(active_tracks):
        for j in range(n_dets):
            d2 = compute_mahalanobis_distance(
                track, detections_z[j], sensor_positions[j],
                sensor_id, coord_manager, vessel_pos,
            )
            if d2 <= GATE_THRESHOLD:
                cost[i, j] = d2

    if not np.any(np.isfinite(cost)):
        return [], list(range(n_tracks)), list(range(n_dets))

    sentinel     = GATE_THRESHOLD * 1e6
    cost_for_lap = np.where(np.isfinite(cost), cost, sentinel)
    row_ind, col_ind = linear_sum_assignment(cost_for_lap)

    matched_pairs:    List[Tuple[int, int]] = []
    matched_track_set: Set[int]             = set()
    matched_det_set:   Set[int]             = set()

    for r, c in zip(row_ind, col_ind):
        if cost[r, c] <= GATE_THRESHOLD:
            matched_pairs.append((int(r), int(c)))
            matched_track_set.add(r)
            matched_det_set.add(c)

    unmatched_tracks     = [i for i in range(n_tracks) if i not in matched_track_set]
    unmatched_detections = [j for j in range(n_dets)   if j not in matched_det_set]

    return matched_pairs, unmatched_tracks, unmatched_detections


# ---------------------------------------------------------------------------
# MultiTargetTracker  (T6 scan loop + T7 lifecycle)
# ---------------------------------------------------------------------------

class MultiTargetTracker:
    """
    Multi-target tracker: GNN association (T6) + track lifecycle (T7).

    One call to process_scan() per sensor per scan time covers:
        predict → convert → GNN associate → update/miss → lifecycle

    Track lifecycle (_manage_track_lifecycle) runs at the end of every
    process_scan() call and handles:
        1. 2-point finite-difference track initiation from unmatched detections
        2. Merging duplicate confirmed tracks
        3. Purging deleted tracks from active_tracks
        4. Rotating unassigned detection buffers for next scan

    Tracks added via add_track() or auto-spawned start as "tentative" and
    transition to "confirmed" after M_CONFIRM hits in N_WINDOW scans.

    Typical usage
    -------------
    coord = CoordinateManager()
    mtt   = MultiTargetTracker(coord)

    # Per-sensor, per-scan-time:
    mtt.process_scan(3.33,  "radar",  radar_meas,  vessel_pos=None)
    mtt.process_scan(3.33,  "camera", camera_meas, vessel_pos=None)
    mtt.process_scan(10.0,  "ais",    ais_meas,    vessel_pos=vpos)
    """

    def __init__(self, coordinate_manager: Any) -> None:
        self.coord = coordinate_manager

        self.active_tracks:             List[Track]           = []
        self.unassigned_detections:     List[Dict[str, Any]]  = []
        self.prev_unassigned_detections: List[Dict[str, Any]] = []

        # Auto-increments for internally spawned tracks.
        # add_track() keeps this consistent with externally assigned IDs.
        self._next_track_id: int = 0

        self.scan_history: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Track management helpers
    # ------------------------------------------------------------------

    def add_track(self, track_id: int, ekf: EKF) -> Track:
        """Register a pre-initialised EKF as a new tentative Track."""
        track = Track(track_id=track_id, ekf=ekf)
        self.active_tracks.append(track)
        # Keep _next_track_id ahead of any externally assigned ID
        self._next_track_id = max(self._next_track_id, track_id + 1)
        return track

    def get_track(self, track_id: int) -> Optional[Track]:
        """Return the Track with the given id, or None."""
        for t in self.active_tracks:
            if t.track_id == track_id:
                return t
        return None

    # ------------------------------------------------------------------
    # Per-scan processing  (T6 entry point, now extended with T7 call)
    # ------------------------------------------------------------------

    def process_scan(
        self,
        scan_time: float,
        sensor_id: str,
        measurements: List[Dict[str, Any]],
        vessel_pos: Optional[np.ndarray],
        is_sensor_available: bool = True,
    ) -> None:
        """
        Process one sensor's detections at a single scan time.

        Steps
        -----
        1. Honour dropout  → return immediately if unavailable
        2. Predict all tracks to scan_time
        3. Convert raw dicts to (z, sensor_pos) via CoordinateManager
        4. GNN association → matched / unmatched_tracks / unmatched_dets
        5. Route:
               matched       → EKF update + mark_matched()
               unmatched trk → mark_missed()
               unmatched det → append to self.unassigned_detections
        6. Track lifecycle  → _manage_track_lifecycle(scan_time)
        """
        if not is_sensor_available:
            return

        # Step 2 — predict
        for track in self.active_tracks:
            track.predict(scan_time)

        # Step 3 — convert measurements
        detections_z:       List[np.ndarray]       = []
        sensor_positions:   List[np.ndarray]        = []
        valid_measurements: List[Dict[str, Any]]   = []

        for meas in measurements:
            try:
                z, _, sensor_pos = self.coord.measurement_to_z(
                    meas, vessel_pos=vessel_pos
                )
            except (ValueError, KeyError):
                continue

            detections_z.append(np.asarray(z, dtype=float).reshape(-1))
            sensor_positions.append(np.asarray(sensor_pos, dtype=float).reshape(2))
            valid_measurements.append(meas)

        # Step 4 — GNN
        matched_pairs, unmatched_tracks, unmatched_dets = associate_gnn(
            active_tracks=self.active_tracks,
            detections_z=detections_z,
            sensor_positions=sensor_positions,
            sensor_id=sensor_id,
            coord_manager=self.coord,
            vessel_pos=vessel_pos,
        )

        # Step 5a — update matched tracks
        for track_idx, det_idx in matched_pairs:
            track      = self.active_tracks[track_idx]
            z          = detections_z[det_idx]
            sensor_pos = sensor_positions[det_idx]

            z_pred = self.coord.h_range_bearing(track.ekf.x, sensor_pos)
            H      = self.coord.H_range_bearing(track.ekf.x, sensor_pos)
            R      = self.coord.R(sensor_id, x_pred=track.ekf.x,
                                  vessel_pos=vessel_pos)

            track.ekf.update(z, z_pred, H, R)
            track.mark_matched()

        # Step 5b — mark unmatched tracks
        for track_idx in unmatched_tracks:
            self.active_tracks[track_idx].mark_missed()

        # Step 5c — collect unmatched detections
        for det_idx in unmatched_dets:
            self.unassigned_detections.append(valid_measurements[det_idx])

        # Step 6 — T7 lifecycle
        self._manage_track_lifecycle(scan_time)

        self.scan_history.append({
            "time": scan_time,
            "tracks": [
                {"id": t.track_id, "pos": list(t.ekf.x.flatten()[:2]), "state": t.state}
                for t in self.active_tracks
            ],
        })

    # ------------------------------------------------------------------
    # T7: Track lifecycle manager
    # ------------------------------------------------------------------

    def _manage_track_lifecycle(self, current_time_s: float) -> None:
        """
        Run the four-phase T7 lifecycle update at the end of every scan.

        Phase 1 — 2-Point Track Initiation
            Pair unmatched detections from the current scan against those
            from the previous scan.  If a pair implies a plausible vessel
            speed (< MAX_SPEED m/s), spawn a new tentative Track using a
            finite-difference velocity estimate.

        Phase 2 — Track Merging
            For every unique pair of confirmed tracks, compute the
            state-space Mahalanobis distance d² = (xi−xj)^T (Pi+Pj)^{-1} (xi−xj).
            If d² < MERGE_THRESHOLD (chi²(4) @ 99 % ≈ 13.28), mark the track
            with the smaller hit_streak as "deleted".

        Phase 3 — Purge Deleted Tracks
            Physically remove any track whose state == "deleted" from
            self.active_tracks.

        Phase 4 — Rotate Unassigned Buffers
            Move self.unassigned_detections → self.prev_unassigned_detections.
            Clear self.unassigned_detections for the next scan.
        """
        # ── Phase 1: 2-point track initiation ─────────────────────────────
        spawned_prev_idx: Set[int] = set()
        spawned_curr_idx: Set[int] = set()

        for j, curr_meas in enumerate(self.unassigned_detections):
            if j in spawned_curr_idx:
                continue

            pos_curr = self._meas_to_ned(curr_meas)
            if pos_curr is None:
                continue
            t_curr = float(curr_meas.get("time", current_time_s))

            for i, prev_meas in enumerate(self.prev_unassigned_detections):
                if i in spawned_prev_idx:
                    continue

                pos_prev = self._meas_to_ned(prev_meas)
                if pos_prev is None:
                    continue
                t_prev = float(prev_meas.get("time", 0.0))

                dt = t_curr - t_prev
                if dt <= 0.0:
                    # Same scan time or out-of-order: cannot infer velocity
                    continue

                d     = float(np.linalg.norm(pos_curr - pos_prev))
                speed = d / dt

                if speed >= MAX_SPEED:
                    continue  # physically implausible

                # ── Spawn a new tentative track ──────────────────────────
                vN = (pos_curr[0] - pos_prev[0]) / dt
                vE = (pos_curr[1] - pos_prev[1]) / dt

                sid = str(curr_meas.get("sensor_id", "radar")).lower()
                # R is 2×2 in measurement space; used as proxy for position
                # and velocity uncertainty per the T7 specification.
                try:
                    R_sensor = self.coord.R(sid)
                except (ValueError, KeyError):
                    R_sensor = np.diag([25.0, 25.0])  # 5 m fallback

                P0 = np.block([
                    [R_sensor,           np.zeros((2, 2))   ],
                    [np.zeros((2, 2)),   R_sensor / dt**2   ],
                ])

                new_ekf   = EKF()
                new_ekf.x = np.array([[pos_curr[0]],
                                      [pos_curr[1]],
                                      [vN],
                                      [vE]], dtype=float)
                new_ekf.P = P0
                new_ekf.t = t_curr

                new_id    = self._next_track_id
                self._next_track_id += 1

                new_track = Track(track_id=new_id, ekf=new_ekf)
                self.active_tracks.append(new_track)

                spawned_prev_idx.add(i)
                spawned_curr_idx.add(j)
                break  # each current detection spawns at most one track

        # Remove consumed detections from the unassigned lists
        self.prev_unassigned_detections = [
            m for i, m in enumerate(self.prev_unassigned_detections)
            if i not in spawned_prev_idx
        ]
        self.unassigned_detections = [
            m for j, m in enumerate(self.unassigned_detections)
            if j not in spawned_curr_idx
        ]

        # ── Phase 2: track merging (confirmed + coasting tracks) ──────────
        confirmed = [t for t in self.active_tracks
                     if t.state in ("confirmed", "coasting")]
        deleted_ids: Set[int] = set()

        for i in range(len(confirmed)):
            ti = confirmed[i]
            if ti.track_id in deleted_ids:
                continue

            for j in range(i + 1, len(confirmed)):
                tj = confirmed[j]
                if tj.track_id in deleted_ids:
                    continue

                xi = np.asarray(ti.ekf.x, dtype=float).flatten()
                xj = np.asarray(tj.ekf.x, dtype=float).flatten()
                S  = ti.ekf.P + tj.ekf.P

                try:
                    diff = xi - xj
                    d2   = float(diff @ np.linalg.inv(S) @ diff)
                except np.linalg.LinAlgError:
                    continue

                if d2 < MERGE_THRESHOLD:
                    # Delete the younger track (smaller hit_streak)
                    if ti.hit_streak >= tj.hit_streak:
                        tj.state = "deleted"
                        deleted_ids.add(tj.track_id)
                    else:
                        ti.state = "deleted"
                        deleted_ids.add(ti.track_id)
                        break  # ti is gone; stop inner loop for ti

        # ── Phase 3: purge deleted tracks ──────────────────────────────────
        self.active_tracks = [t for t in self.active_tracks
                              if t.state != "deleted"]

        # ── Phase 4: rotate unassigned detection buffers ───────────────────
        self.prev_unassigned_detections = list(self.unassigned_detections)
        self.unassigned_detections      = []

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _meas_to_ned(self, meas: Dict[str, Any]) -> Optional[np.ndarray]:
        """
        Extract a 2-D NED position [N, E] from a raw measurement dict.

        - radar / camera: back-project (range, bearing) through the sensor origin
        - ais:            read north_m / east_m directly (absolute NED)

        Returns None if the measurement has insufficient fields.
        """
        sid = str(meas.get("sensor_id", "")).lower()

        if sid in ("radar", "camera"):
            r = meas.get("range_m")
            b = meas.get("bearing_rad")
            if r is None or b is None:
                return None
            return self.coord.ned_from_range_bearing(
                float(r), float(b), sensor_id=sid
            )

        if sid == "ais":
            n = meas.get("north_m")
            e = meas.get("east_m")
            if n is None or e is None:
                return None
            return np.array([float(n), float(e)], dtype=float)

        return None

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Human-readable state snapshot for debugging."""
        n_tent  = sum(1 for t in self.active_tracks if t.state == "tentative")
        n_conf  = sum(1 for t in self.active_tracks if t.state == "confirmed")
        n_coast = sum(1 for t in self.active_tracks if t.state == "coasting")
        lines   = [
            f"MultiTargetTracker — {len(self.active_tracks)} active "
            f"({n_conf} confirmed, {n_coast} coasting, {n_tent} tentative) "
            f"| GNN gate={GATE_THRESHOLD:.2f} | merge gate={MERGE_THRESHOLD:.2f}"
        ]
        for t in self.active_tracks:
            lines.append(f"  {t}")
        lines.append(
            f"  Prev-unassigned buffer: {len(self.prev_unassigned_detections)}"
        )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from coordinate_manager import CoordinateManager

    coord = CoordinateManager()
    print(f"GNN gate    chi²(df={_DOF_MEAS}, p={_GATE_PROB}) = {GATE_THRESHOLD:.4f}")
    print(f"Merge gate  chi²(df={_DOF_STATE}, p=0.99)        = {MERGE_THRESHOLD:.4f}")

    # ── 1. Track state machine ─────────────────────────────────────────────
    print("\n── Track state machine ──")
    ekf0 = EKF()
    ekf0.x = np.array([[500.], [300.], [-2.], [-1.]])
    ekf0.t = 0.0
    t0 = Track(track_id=0, ekf=ekf0)
    assert t0.state == "tentative"

    for k in range(M_CONFIRM):
        t0.mark_matched()
    assert t0.state == "confirmed", f"Expected confirmed after {M_CONFIRM} hits"
    print(f"Confirmed after {M_CONFIRM} hits: {t0.state} ✓")

    for _ in range(K_DEL):
        t0.mark_missed()
    assert t0.state == "deleted", f"Expected deleted after {K_DEL} misses"
    print(f"Deleted after {K_DEL} misses:     {t0.state} ✓")

    # ── 2. Sliding window — tentative track with miss in window ───────────
    print("\n── Sliding window (M=3, N=5) ──")
    ekf1 = EKF(); ekf1.x = np.array([[200.], [400.], [1.], [2.]]); ekf1.t = 0.0
    t1 = Track(track_id=1, ekf=ekf1)
    # H M H H  → sum = 3 within window of 4, should confirm on 4th hit
    for h in [True, False, True, True]:
        if h:
            t1.mark_matched()
        else:
            t1.mark_missed()
    assert t1.state == "confirmed", f"Unexpected state: {t1.state}"
    print(f"Confirmed with H/M/H/H pattern:   {t1.state} ✓")

    # ── 3. Mahalanobis distance ────────────────────────────────────────────
    print("\n── Mahalanobis distance ──")
    ekf2 = EKF(); ekf2.x = np.array([[493.3], [296.7], [-2.0], [-1.0]]); ekf2.t = 0.0
    t2 = Track(track_id=2, ekf=ekf2)
    z_near = coord.h_range_bearing(ekf2.x, coord.sensor_position("radar"))
    d2_near = compute_mahalanobis_distance(t2, z_near, coord.sensor_position("radar"),
                                           "radar", coord)
    assert d2_near < GATE_THRESHOLD
    print(f"d² (near, expect < {GATE_THRESHOLD:.2f}): {d2_near:.4f} ✓")

    z_far = np.array([950.0, 1.5])
    d2_far = compute_mahalanobis_distance(t2, z_far, coord.sensor_position("radar"),
                                          "radar", coord)
    assert d2_far > GATE_THRESHOLD
    print(f"d² (far,  expect > {GATE_THRESHOLD:.2f}): {d2_far:.2f} ✓")

    # ── 4. GNN association ────────────────────────────────────────────────
    print("\n── GNN association (2 tracks, 2 in-gate dets) ──")
    tracks = [t1, t2]
    z0 = coord.h_range_bearing(t1.ekf.x, coord.sensor_position("radar"))
    z1 = coord.h_range_bearing(t2.ekf.x, coord.sensor_position("radar"))
    rpos = coord.sensor_position("radar")
    matched, u_trk, u_det = associate_gnn(tracks, [z0, z1], [rpos, rpos], "radar", coord)
    assert len(matched) == 2 and len(u_trk) == 0 and len(u_det) == 0
    print(f"matched={matched}, u_trk={u_trk}, u_det={u_det} ✓")

    # ── 5. 2-point track initiation ───────────────────────────────────────
    print("\n── 2-point initiation ──")
    mtt = MultiTargetTracker(coord)

    # Scan 1: one unmatched radar detection
    det1 = {"sensor_id": "radar", "time": 0.0,
             "range_m": 500.0, "bearing_rad": 0.6435,    # ≈ NE
             "is_false_alarm": False, "target_id": -1}
    mtt.process_scan(0.0, "radar", [det1], vessel_pos=None)
    assert len(mtt.active_tracks) == 0         # not yet paired
    assert len(mtt.prev_unassigned_detections) == 1
    print(f"After scan 1: active={len(mtt.active_tracks)}, "
          f"prev_buf={len(mtt.prev_unassigned_detections)} ✓")

    # Scan 2: same target has moved ~5 m in 3.33 s (≈ 1.5 m/s < 15 m/s)
    import math as _math
    pos1_ned = coord.ned_from_range_bearing(500.0, 0.6435, "radar")
    pos2_ned = pos1_ned + np.array([5.0, 3.0])
    r2 = float(np.linalg.norm(pos2_ned))
    b2 = float(_math.atan2(pos2_ned[1], pos2_ned[0]))
    det2 = {"sensor_id": "radar", "time": 3.333,
             "range_m": r2, "bearing_rad": b2,
             "is_false_alarm": False, "target_id": -1}
    mtt.process_scan(3.333, "radar", [det2], vessel_pos=None)
    assert len(mtt.active_tracks) == 1, f"Expected 1 spawned track, got {len(mtt.active_tracks)}"
    spawned = mtt.active_tracks[0]
    assert spawned.state == "tentative"
    print(f"After scan 2: active={len(mtt.active_tracks)}, "
          f"state={spawned.state} ✓")
    print(f"  Spawned: {spawned}")

    # ── 6. Track merging ──────────────────────────────────────────────────
    print("\n── Track merging (duplicate confirmed tracks) ──")
    mtt2  = MultiTargetTracker(coord)
    pos   = np.array([400.0, 300.0])

    for tid in (10, 11):
        e = EKF()
        e.x = np.array([[pos[0]], [pos[1]], [0.0], [0.0]])
        e.P = np.diag([100.0, 100.0, 25.0, 25.0])
        e.t = 0.0
        tr  = mtt2.add_track(tid, e)
        # Force into confirmed state
        tr.state     = "confirmed"
        tr.hit_streak = 5 if tid == 10 else 3

    assert len(mtt2.active_tracks) == 2
    mtt2._manage_track_lifecycle(0.0)
    assert len(mtt2.active_tracks) == 1, \
        f"Expected merge to 1 track, got {len(mtt2.active_tracks)}"
    survivor = mtt2.active_tracks[0]
    assert survivor.track_id == 10, "Older track (id=10, more hits) should survive"
    print(f"Merged: survivor is track {survivor.track_id} "
          f"(hit_streak={survivor.hit_streak}) ✓")

    # ── 7. K_DEL deletion via process_scan ───────────────────────────────
    print("\n── K_DEL deletion via process_scan ──")
    mtt3 = MultiTargetTracker(coord)
    e3   = EKF(); e3.x = np.array([[300.], [200.], [0.], [0.]]); e3.t = 0.0
    mtt3.add_track(20, e3)

    for k in range(K_DEL):
        mtt3.process_scan(float(k) * 3.33, "radar", [], vessel_pos=None)

    assert len(mtt3.active_tracks) == 0, \
        f"Expected track purged after {K_DEL} misses; got {len(mtt3.active_tracks)}"
    print(f"Track purged after {K_DEL} consecutive misses ✓")

    print("\n══ All self-tests passed ══")
