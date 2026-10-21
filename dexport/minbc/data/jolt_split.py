"""Quest tracking jumps -> extra episode boundaries (``data.jolt_split``).

The Quest's tracking occasionally loses a controller and snaps it back a few centimetres away, a
one-frame step that does not return. With relative arm actions that step becomes one huge action
label, so each detected step starts a new episode segment; no frame is removed or modified.
With ``d[i] = |pos[i+1] - pos[i]|`` and ``z = d / p95(d)`` per trajectory and controller, a step
fires if ``(d >= floor_m and z >= z_thresh) or d >= abs_gate_m``: fast hand motion passes the
floor and tremor passes the z test, but not both; the absolute gate catches trajectories whose
p95 is inflated by many jumps. ``abs_gate_m`` can clip genuinely ballistic motion.
"""

from typing import List, Sequence, Tuple

import numpy as np

from dexport.minbc.config import JOLT_ABS_GATE_M, JOLT_FLOOR_M, JOLT_Z_THRESH


def detect_steps(
    pos: np.ndarray,
    z_thresh: float = JOLT_Z_THRESH,
    floor_m: float = JOLT_FLOOR_M,
    abs_gate_m: float = JOLT_ABS_GATE_M,
) -> np.ndarray:
    """Bool mask over the ``T - 1`` steps of one (T, 3) position stream (True = a jump).

    NaN steps never fire, so a controller that was never tracked yields no cuts.
    """
    pos = np.asarray(pos, dtype=np.float64)
    if pos.ndim != 2 or pos.shape[0] < 2 or pos.shape[1] < 3:
        return np.zeros(max(0, pos.shape[0] - 1), dtype=bool)
    d = np.linalg.norm(np.diff(pos[:, :3], axis=0), axis=1)
    finite = d[np.isfinite(d)]
    if len(finite) < 5:
        return np.zeros(len(d), dtype=bool)
    q = max(float(np.percentile(finite, 95)), 1e-9)
    with np.errstate(invalid="ignore"):
        fire = ((d >= floor_m) & ((d / q) >= z_thresh)) | (d >= abs_gate_m)
    return np.nan_to_num(fire, nan=False).astype(bool)


def find_cuts(
    pose_streams: Sequence[np.ndarray],
    T: int,
    z_thresh: float = JOLT_Z_THRESH,
    floor_m: float = JOLT_FLOOR_M,
    abs_gate_m: float = JOLT_ABS_GATE_M,
) -> List[int]:
    """Frames in ``(0, T)`` that start a new segment, over every controller.

    The union over controllers: for a bimanual policy a jump of either controller corrupts the
    window's action label.
    """
    cuts: set = set()
    for pos in pose_streams:
        pos = np.asarray(pos)
        if pos.ndim != 2 or pos.shape[0] < 2:
            continue
        mask = detect_steps(pos[:T], z_thresh, floor_m, abs_gate_m)
        cuts |= set((np.where(mask)[0] + 1).tolist())
    return sorted(c for c in cuts if 0 < c < T)


def split_lengths(cuts: Sequence[int], T: int, min_len: int) -> Tuple[List[int], int]:
    """``(segment_lengths, n_runt_frames)`` for cut offsets in a trajectory of length ``T``.

    Every segment is returned, also those shorter than ``min_len``, so every buffer row stays in
    an episode. ``n_runt_frames`` counts the frames of segments shorter than ``min_len`` (the
    window length), although segments down to ``window_len - pad_before - pad_after`` frames
    still form edge-padded windows.
    """
    bounds = [0] + [int(c) for c in cuts if 0 < c < T] + [int(T)]
    lengths = [
        bounds[i + 1] - bounds[i] for i in range(len(bounds) - 1) if bounds[i + 1] > bounds[i]
    ]
    runt = sum(L for L in lengths if L < min_len)
    return lengths, runt
