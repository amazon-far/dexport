"""Detect hand readback channels that never move although the hand was commanded to.

A dead readback records one snapshot for the whole episode: non-NaN, the right length, a plausible
near-zero pose, so it passes every other check and would train as a constant proprio observation.
A constant channel alone is not a fault (the unused hand of a one-hand session is constant), so
each measured channel is judged against its own commanded channel (:func:`diagnose_hand_channels`).
Pure numpy, shared by the live recorder (:mod:`dexport.collect.recorder`) and the offline scanner
(``scripts/check_frozen_channels.py``).
"""

from __future__ import annotations

import warnings

import numpy as np

from dexport.common.channels import CHANNELS

# A joint is "frozen" if its range over the episode is below this (rad): three orders of magnitude
# under the encoder noise of a still hand (~1e-3 rad per frame), while a held snapshot is
# bit-identical and has a range of exactly 0.
FROZEN_EPS_RAD = 1e-6

# (commanded channel, measured channel, human label) for each hand.
HAND_CHANNEL_PAIRS = tuple(
    (ch.command, ch.measured, f"{side.capitalize()} hand") for side, ch in CHANNELS.items()
)


def _per_joint_range(a: np.ndarray) -> np.ndarray:
    """``nanmax - nanmin`` over time of a (T, n) array; NaN for a joint that is never finite."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # "All-NaN slice encountered"
        return np.nanmax(a, axis=0) - np.nanmin(a, axis=0)


def channel_span(arr) -> float:
    """Largest per-element range over time, NaN-safe; NaN if nothing is finite.

    Per element (per joint) rather than a norm, so a single stuck joint is not masked by its
    neighbours moving; ``nanmax - nanmin`` rather than a difference to row 0, so a leading NaN
    placeholder (recorded until a side's first real sample) does not poison the result.
    """
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim < 1 or a.shape[0] == 0:
        return float("nan")
    a = a.reshape(a.shape[0], -1)
    if not np.any(np.isfinite(a)):
        return float("nan")
    rng = _per_joint_range(a)
    if not np.any(np.isfinite(rng)):
        return float("nan")
    return float(np.nanmax(rng))


def frozen_joint_count(arr) -> int | None:
    """How many of the individual joints never move; None if nothing is finite. A partial count
    is the signature of a single dead joint stuck at 0 (rather than the whole readback held), so
    it is reported alongside the whole-channel verdict."""
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim < 1 or a.shape[0] == 0 or not np.any(np.isfinite(a)):
        return None
    rng = _per_joint_range(a.reshape(a.shape[0], -1))
    return int(np.sum(np.nan_to_num(rng, nan=np.inf) < FROZEN_EPS_RAD))


def diagnose_hand_channels(
    channels: dict[str, np.ndarray],
    eps: float = FROZEN_EPS_RAD,
) -> list[dict]:
    """Judge each hand's measured channel against its commanded channel:

        commanded   measured        verdict
        moves       moves           healthy (no finding)
        moves       frozen          readback dead (warn)
        moves       some joints     stuck joints (warn)
        frozen      frozen          hand not driven (info)
        frozen      moves           moved without a command (info)

    ``channels`` maps channel names to (T, ...) arrays; a hand whose channels are missing or
    all-NaN is skipped (an all-NaN hand was never driven, which the recorder reports itself).
    Returns the findings, each ``{level, label, cmd_key, meas_key, cmd_span, meas_span,
    frozen_joints, n_joints, message}`` with ``level`` "warn" or "info"; an empty list means every
    driven hand reported motion.
    """
    findings: list[dict] = []

    for cmd_key, meas_key, label in HAND_CHANNEL_PAIRS:
        if cmd_key not in channels or meas_key not in channels:
            continue
        cmd_span = channel_span(channels[cmd_key])
        meas_span = channel_span(channels[meas_key])
        # Never driven / never reported at all -> not this check's business.
        if not np.isfinite(cmd_span) or not np.isfinite(meas_span):
            continue

        n_frozen = frozen_joint_count(channels[meas_key])
        meas_arr = np.asarray(channels[meas_key])
        n_joints = int(np.prod(meas_arr.shape[1:])) if meas_arr.ndim > 1 else 1

        cmd_moved = cmd_span >= eps
        meas_moved = meas_span >= eps

        if cmd_moved and not meas_moved:
            level = "warn"
            message = (
                f"{label} READBACK DEAD: {meas_key} never changed "
                f"(range {meas_span:.2e} rad over all {n_joints} joints) while "
                f"{cmd_key} was commanded {cmd_span:.3f} rad. The measured "
                f"channel is holding one snapshot -- check the hand's connection "
                f"(USB or Ethernet) before collecting more. This trains as a "
                f"CONSTANT proprio observation."
            )
        elif cmd_moved and n_frozen:
            level = "warn"
            message = (
                f"{label} has {n_frozen}/{n_joints} STUCK joint(s) in "
                f"{meas_key} (range < {eps:.0e} rad) while {cmd_key} moved "
                f"{cmd_span:.3f} rad."
            )
        elif not cmd_moved and not meas_moved:
            level = "info"
            message = (
                f"{label} not driven this trajectory ({cmd_key} and {meas_key} "
                f"both constant) -- expected for the unused hand in a "
                f"single-hand session."
            )
        elif not cmd_moved and meas_moved:
            level = "info"
            message = (
                f"{label} moved {meas_span:.3f} rad with no command "
                f"({cmd_key} constant) -- backdriven or driven outside the "
                f"recorded action."
            )
        else:
            continue  # commanded and measured both moved: healthy, stay quiet.

        findings.append(
            {
                "level": level,
                "label": label,
                "cmd_key": cmd_key,
                "meas_key": meas_key,
                "cmd_span": cmd_span,
                "meas_span": meas_span,
                "frozen_joints": n_frozen,
                "n_joints": n_joints,
                "message": message,
            }
        )

    return findings
