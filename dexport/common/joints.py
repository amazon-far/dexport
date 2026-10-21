"""Wuji hand joint conventions, shared by the drivers, the skills, deploy and the Isaac Lab env.

Two orders of the 20 finger joints: MuJoCo / finger-major ``left_finger{1..5}_joint{1..4}`` (the
row-major flatten of the hand's ``(5, 4)`` grid, rows thumb..pinky; drivers and diffusion policy)
and Isaac Lab articulation / joint-major ``[f1j1, f2j1, ..., f5j1, f1j2, ..., f5j4]`` (the rotation
policy). Mixing them moves the wrong joints without any error, so every crossing goes through
:func:`mujoco_to_articulation` / :func:`articulation_to_mujoco`. Also holds the Wuji 1 constants
(thumb-base neutral, URDF joint limits). Numpy only: the Isaac Lab env imports this module too.
"""

from __future__ import annotations

import numpy as np

MUJOCO_ORDER: list[str] = [f"left_finger{i}_joint{j}" for i in range(1, 6) for j in range(1, 5)]
ARTICULATION_ORDER: list[str] = [
    f"left_finger{i}_joint{j}" for j in range(1, 5) for i in range(1, 6)
]

_ART_TO_MJ = np.array([MUJOCO_ORDER.index(n) for n in ARTICULATION_ORDER], dtype=np.int64)
_MJ_TO_ART = np.array([ARTICULATION_ORDER.index(n) for n in MUJOCO_ORDER], dtype=np.int64)
assert sorted(ARTICULATION_ORDER) == sorted(MUJOCO_ORDER)
assert np.array_equal(_ART_TO_MJ[_MJ_TO_ART], np.arange(20))
assert np.array_equal(_MJ_TO_ART[_ART_TO_MJ], np.arange(20))

# MuJoCo-order indices of the finger 2-5 abduction joints, the only joints whose sign convention
# differs between the left and right Wuji 1 hands.
_ABDUCTION_MJ = [MUJOCO_ORDER.index(f"left_finger{f}_joint2") for f in (2, 3, 4, 5)]

# Wuji 1 neutral of the thumb base (finger1_joint1): its lower limit is +0.0475 rad, so commanding 0
# would drive it into the stop. Every other neutral joint angle is 0.
WUJI1_THUMB_BASE_NEUTRAL = 0.098

# Joint limits (lower, upper) of the Wuji 1 hand from its URDF, MuJoCo joint names (rad). The soft
# sim URDF limits the finger 2-5 abduction joints (joint2) to ±0.37 instead.
WUJI1_JOINT_LIMITS = {
    "left_finger1_joint1": (0.0475, 1.6033),
    "left_finger1_joint2": (-0.1387, 0.9324),
    "left_finger1_joint3": (-0.4642, 1.5623),
    "left_finger1_joint4": (-0.4699, 1.5568),
    "left_finger2_joint1": (-0.1585, 1.5604),
    "left_finger2_joint2": (-0.4069, 0.2988),
    "left_finger2_joint3": (-0.4777, 1.5485),
    "left_finger2_joint4": (-0.4683, 1.5753),
    "left_finger3_joint1": (-0.1644, 1.5516),
    "left_finger3_joint2": (-0.3995, 0.3055),
    "left_finger3_joint3": (-0.4739, 1.5512),
    "left_finger3_joint4": (-0.4684, 1.5745),
    "left_finger4_joint1": (-0.1554, 1.5585),
    "left_finger4_joint2": (-0.4118, 0.3087),
    "left_finger4_joint3": (-0.4765, 1.5487),
    "left_finger4_joint4": (-0.4777, 1.5634),
    "left_finger5_joint1": (-0.1626, 1.5585),
    "left_finger5_joint2": (-0.4153, 0.2940),
    "left_finger5_joint3": (-0.4768, 1.5490),
    "left_finger5_joint4": (-0.4683, 1.5735),
}


def articulation_to_mujoco(vec_art: np.ndarray) -> np.ndarray:
    """Reorder a 20-vector from articulation order to MuJoCo order (keeps the dtype)."""
    vec_art = np.asarray(vec_art).reshape(20)
    out = np.empty(20, dtype=vec_art.dtype)
    out[_ART_TO_MJ] = vec_art
    return out


def mujoco_to_articulation(vec_mj: np.ndarray) -> np.ndarray:
    """Reorder a 20-vector from MuJoCo order to articulation order (keeps the dtype)."""
    vec_mj = np.asarray(vec_mj).reshape(20)
    out = np.empty(20, dtype=vec_mj.dtype)
    out[_MJ_TO_ART] = vec_mj
    return out


def mirror_abduction(q: np.ndarray) -> np.ndarray:
    """Map a Wuji 1 pose between the left- and right-hand conventions.

    Negates the abduction joints of fingers 2-5 (``finger{2..5}_joint2``). Takes a ``(5, 4)`` grid
    or a MuJoCo-order 20-vector and returns a float64 copy of the same shape. The map is its own
    inverse, so it also converts right-hand readings back to the left convention.
    """
    out = np.array(q, dtype=np.float64, order="C", copy=True)
    if out.size != 20:
        raise ValueError(f"expected a (5, 4) grid or a 20-vector, got shape {out.shape}")
    flat = out.reshape(-1)
    flat[_ABDUCTION_MJ] = -flat[_ABDUCTION_MJ]
    return out
