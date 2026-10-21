"""Observation and action normalization, shared by the training Dataset and inference
(p = ``data.norm_percentile``, 2).

- ``stats.pkl``: ``action`` (merged, arm blocks first), ``arm_action[_r]``, ``ee_pos_rel[_r]``
  and ``ee_quat_rel[_r]``. The arm actions and the relative end-effector observations are bounded
  by their [p, 100 - p] percentiles (rot6d entries fixed to [-1, 1]), the hand action targets by
  their min/max; both map linearly to [-1, 1] and are clipped at ±1 (the hand targets only in the
  train split the bounds were fitted on, see ``data/dataset.py``).
- ``norm.pkl``: ``{key: {"lower", "upper"}}``, the [p, 100 - p] percentiles of every other
  proprio observation, which is clipped at ±1.5 (±1 when p = 0).
"""

import numpy as np

#: Relative end-effector observation key -> its min/max entry in stats.pkl.
REL_EE_STATS_KEYS = {
    "ee_pos": "ee_pos_rel",
    "ee_quat": "ee_quat_rel",
    "ee_pos_r": "ee_pos_rel_r",
    "ee_quat_r": "ee_quat_rel_r",
}


def minmax_norm_data(data, dmin, dmax):
    """Map ``[dmin, dmax]`` to ``[-1, 1]``; the 1e-8 keeps constant dims finite."""
    ndata = (data - dmin) / ((dmax - dmin) + 1e-8)
    return ndata * 2 - 1


def minmax_unnorm_data(ndata, dmin, dmax):
    """Inverse of :func:`minmax_norm_data`."""
    ndata = (ndata + 1) / 2
    return ndata * (dmax - dmin + 1e-8) + dmin


def action_bounds(stats, action_dim):
    """``(min, max)`` of the merged action stats; their width must be ``action_dim``."""
    dmin, dmax = np.asarray(stats["action"]["min"]), np.asarray(stats["action"]["max"])
    if dmin.shape != (action_dim,) or dmax.shape != (action_dim,):
        raise ValueError(
            f"stats['action'] bounds have shape {dmin.shape} / {dmax.shape} but the policy has "
            f"{action_dim} action dims (stats.pkl of another run?)"
        )
    return dmin, dmax


def minmax_bounds(x):
    """Per-dim ``{"min", "max"}`` over every leading axis; values beyond 1e5 mean corrupt data."""
    x = x.reshape(-1, x.shape[-1])
    bounds = {"min": np.min(x, axis=0), "max": np.max(x, axis=0)}
    if np.any(bounds["max"] > 1e5) or np.any(bounds["min"] < -1e5):
        raise ValueError("data out of range")
    return bounds


def percentile_bounds(x, percentile):
    """Per-dim ``[p, 100 - p]`` percentile bounds; ``p <= 0`` gives the true min/max."""
    x = x.reshape(-1, x.shape[-1])
    if percentile and percentile > 0:
        return {
            "min": np.percentile(x, percentile, axis=0),
            "max": np.percentile(x, 100.0 - percentile, axis=0),
        }
    return {"min": np.min(x, axis=0), "max": np.max(x, axis=0)}


def set_identity_rot6d(bounds, lo, hi):
    """Set dims ``[lo, hi)`` of min/max bounds to (-1, 1), in place.

    rot6d entries are rotation-matrix entries, already in [-1, 1]; identity bounds keep the
    rotation geometry instead of stretching each axis by its own percentiles.
    """
    bounds["min"] = np.asarray(bounds["min"], dtype=np.float64).copy()
    bounds["max"] = np.asarray(bounds["max"], dtype=np.float64).copy()
    bounds["min"][lo:hi] = -1.0
    bounds["max"][lo:hi] = 1.0
    return bounds


def obs_clip_limit(norm_percentile):
    """Clip for percentile-normalized observations: 1.5 keeps tail headroom, min/max clips at 1."""
    return 1.5 if (norm_percentile and norm_percentile > 0) else 1.0


def normalize_percentile(x, lower, upper, clip):
    """Map ``[lower, upper]`` to ``[-1, 1]`` (constant dims get span 1) and clip to ``±clip``."""
    mid = 0.5 * (lower + upper)
    span = upper - lower
    span_safe = np.where(span < 1e-12, 1.0, span)
    return np.clip(2.0 * (x - mid) / span_safe, -clip, clip)


def normalize_obs(key, x, stats, obs_norm, clip):
    """Normalize one proprio observation stream exactly as the policy was trained on it."""
    if key in REL_EE_STATS_KEYS:
        bounds = stats[REL_EE_STATS_KEYS[key]]
        return np.clip(minmax_norm_data(x, bounds["min"], bounds["max"]), -1, 1)
    bounds = obs_norm[key]
    return normalize_percentile(x, bounds["lower"], bounds["upper"], clip)
