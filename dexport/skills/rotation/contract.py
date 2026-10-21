"""Interface of the in-hand rotation skill, shared by the Isaac Lab env (``sim/``)
and the hardware runner (:mod:`dexport.skills.rotation.runner`).

Observation (``OBS_DIM`` = 147, float32, every term clipped to ``+-OBS_CLIP``): ``[joint position
history | joint target history | hand root position (3) | hand root quaternion wxyz (4) | previous
action (20)]``, each history ``HISTORY_LENGTH`` steps, oldest first. Action (20-D): clipped to
``+-ACTION_CLIP``, ring and pinky zeroed, then accumulated as ``target += ACTION_SCALE * a`` and
clamped to :func:`joint_limits_art`. Joint vectors are in articulation order and the LEFT-hand
convention (:mod:`dexport.common.joints`). Imports only numpy and :mod:`dexport.common`.
"""

from __future__ import annotations

import numpy as np

from dexport.common.joints import (
    ARTICULATION_ORDER,
    MUJOCO_ORDER,
    WUJI1_JOINT_LIMITS,
    mujoco_to_articulation,
)

POLICY_HZ = 20.0  # sim control rate: 200 Hz physics with decimation 10
HISTORY_LENGTH = 3
NUM_JOINTS = 20
OBS_CLIP = 5.0
ACTION_SCALE = 0.04
ACTION_CLIP = 1.0
FIXED_FINGERS = ("finger4", "finger5")  # ring and pinky: masked out of the action
CRITIC_OBS_DIM = 127  # privileged critic observation (sim only; part of the checkpoint format)


def obs_dim(history_length: int = HISTORY_LENGTH) -> int:
    """Actor observation size for a history length (147 for the default 3)."""
    return 2 * history_length * NUM_JOINTS + 3 + 4 + NUM_JOINTS


OBS_DIM = obs_dim()
_ROOT = 2 * HISTORY_LENGTH * NUM_JOINTS
ROOT_POS_SLICE = slice(_ROOT, _ROOT + 3)
ROOT_QUAT_SLICE = slice(_ROOT + 3, _ROOT + 7)

# Hand root pose in the sim (the hand base is fixed): position (m), and orientation (wxyz) = the
# env's base hand orientation (0.5, -0.5, 0.5, 0.5) with the palm pitched a further 45 deg down
# toward the handle. The policy observes it as a constant, so a checkpoint with an observation
# normalizer carries the training value in the normalizer mean at ROOT_POS_SLICE / ROOT_QUAT_SLICE
# (zero variance); the runner reads it from there and falls back to these values.
HAND_ROOT_POS = (-0.0014, 0.0154, 0.2105)
HAND_ROOT_QUAT_WXYZ = (0.270598, -0.653281, 0.653281, 0.270598)

# Start grasp of every rotation episode (sim reset and hardware start), MuJoCo joint names (rad).
INIT_GRASP = {
    "left_finger1_joint1": 1.1500,
    "left_finger1_joint2": 0.0010,
    "left_finger1_joint3": 0.5332,
    "left_finger1_joint4": -0.1400,
    "left_finger2_joint1": 0.2960,
    "left_finger2_joint2": -0.2700,
    "left_finger2_joint3": 1.1320,
    "left_finger2_joint4": 0.1601,
    "left_finger3_joint1": 0.5860,
    "left_finger3_joint2": 0.0000,
    "left_finger3_joint3": 0.6311,
    "left_finger3_joint4": 0.3405,
    "left_finger4_joint1": 0.6937,
    "left_finger4_joint2": -0.0697,
    "left_finger4_joint3": 0.9670,
    "left_finger4_joint4": 0.3981,
    "left_finger5_joint1": 0.5122,
    "left_finger5_joint2": 0.0840,
    "left_finger5_joint3": 1.5485,
    "left_finger5_joint4": 0.5221,
}

# Joint limits of the real hand, as collected: the Wuji 1 plain-URDF limits. Sim training used the
# soft URDF, which limits the finger 2-5 abduction joints (joint2) to ±0.37. Of those bounds only
# the lower one of left_finger2_joint2 (-0.37 in sim, -0.4069 here) reaches the target clamp:
# POLICY_JOINT_LIMITS replaces the others of fingers 2-3, and fingers 4-5 are masked out of the
# action.
HARDWARE_JOINT_LIMITS = WUJI1_JOINT_LIMITS

# Tighter limits the policy was trained with (sim and hardware); None keeps the URDF bound.
POLICY_JOINT_LIMITS = {
    "left_finger1_joint1": (0.9, None),
    "left_finger2_joint1": (0.25, 1.25),
    "left_finger2_joint2": (None, 0.1),
    "left_finger2_joint3": (-0.35, 1.20),
    "left_finger2_joint4": (-0.25, 1.25),
    "left_finger3_joint1": (0.3, 1.25),
    "left_finger3_joint2": (-0.25, 0.05),
    "left_finger3_joint3": (-0.35, 1.0),
    "left_finger3_joint4": (-0.25, 1.25),
}


def joint_limits_art() -> tuple[np.ndarray, np.ndarray]:
    """``(lower, upper)`` target clamp in articulation order: the hardware limits tightened by
    :data:`POLICY_JOINT_LIMITS`."""
    lo = np.array([HARDWARE_JOINT_LIMITS[n][0] for n in MUJOCO_ORDER], dtype=np.float64)
    hi = np.array([HARDWARE_JOINT_LIMITS[n][1] for n in MUJOCO_ORDER], dtype=np.float64)
    for name, (p_lo, p_hi) in POLICY_JOINT_LIMITS.items():
        k = MUJOCO_ORDER.index(name)
        if p_lo is not None:
            lo[k] = p_lo
        if p_hi is not None:
            hi[k] = p_hi
    return mujoco_to_articulation(lo), mujoco_to_articulation(hi)


def init_grasp_art() -> np.ndarray:
    """:data:`INIT_GRASP` in articulation order, clamped to :func:`joint_limits_art`."""
    lo, hi = joint_limits_art()
    grasp = np.array([INIT_GRASP[n] for n in MUJOCO_ORDER], dtype=np.float64)
    return np.clip(mujoco_to_articulation(grasp), lo, hi)


def action_mask_art() -> np.ndarray:
    """Per-joint action mask in articulation order: 0 for the :data:`FIXED_FINGERS` joints."""
    return np.array(
        [0.0 if any(f in name for f in FIXED_FINGERS) else 1.0 for name in ARTICULATION_ORDER],
        dtype=np.float64,
    )
