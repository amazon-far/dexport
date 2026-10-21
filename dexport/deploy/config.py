"""Rig configuration for deploying on the Tianji arms with Wuji hands: the calibrated mounts
between the Quest controller and each flange, the hand's reset ramp and the deploy loop.

The mount of an arm, ``M = T_hand_to_flange · T_quest_to_hand`` (:func:`mount_transform`), is the
controller frame in flange coordinates: a fitted Quest -> hand transform per (hand model, side)
(``QUEST_TO_HAND``, fitted by :mod:`dexport.calib.solve`) and the hand -> flange bracket, the same
on both flanges. The fits are stored as calibrated, i.e. paired with the standoff of the bracket
with the controller bracket's 12 mm spacer installed, so the deploy-time ``--spacer`` choice
changes only the standoff (:func:`hand_to_flange_t`). The arm driver's settings are
:class:`dexport.hardware.tianji.arms.TianjiConfig`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from dexport.common.pose import make_transform
from dexport.hardware.tianji.arms import TianjiConfig


def _orthonormalize(R) -> np.ndarray:
    """The nearest rotation to ``R`` (by SVD): removes the rounding of a fitted matrix written
    as decimals."""
    U, _, Vt = np.linalg.svd(np.asarray(R, dtype=np.float64))
    if np.linalg.det(U @ Vt) < 0:
        U = U @ np.diag([1.0, 1.0, -1.0])
    return U @ Vt


# Quest -> hand fits (translation in m, rotation) of ``python -m dexport.calib.solve``: Wuji 1 from
# pooled capture sessions, Wuji 2 from one session per side. The two sides are independent fits,
# not mirror images.
QUEST_TO_LH_T: np.ndarray = np.array([-0.028583, -0.044385, 0.083733], dtype=np.float64)
QUEST_TO_LH_R: np.ndarray = _orthonormalize(
    np.array(
        [
            [-0.99623249, -0.08583906, -0.01234803],
            [-0.08255283, 0.89504885, 0.43826087],
            [-0.02656782, 0.43762909, -0.89876300],
        ],
        dtype=np.float64,
    )
)

QUEST_TO_RH_T: np.ndarray = np.array([0.032546, -0.047614, 0.084178], dtype=np.float64)
QUEST_TO_RH_R: np.ndarray = _orthonormalize(
    np.array(
        [
            [-0.995943455, +0.089083599, -0.012678590],
            [+0.075188728, +0.901313972, +0.426590881],
            [+0.049429641, +0.423907109, -0.904355833],
        ],
        dtype=np.float64,
    )
)

QUEST_TO_LH_T_V2: np.ndarray = np.array([-0.030639, -0.042971, 0.087934], dtype=np.float64)
QUEST_TO_LH_R_V2: np.ndarray = _orthonormalize(
    np.array(
        [
            [-0.996382102, -0.084986512, -0.000012702],
            [-0.076461719, +0.896372148, +0.436658423],
            [-0.037098691, +0.435079609, -0.899627379],
        ],
        dtype=np.float64,
    )
)

QUEST_TO_RH_T_V2: np.ndarray = np.array([0.040387, -0.037306, 0.084081], dtype=np.float64)
QUEST_TO_RH_R_V2: np.ndarray = _orthonormalize(
    np.array(
        [
            [-0.996568667, +0.081753804, -0.012930892],
            [+0.068248843, +0.900029062, +0.430452997],
            [+0.046829348, +0.428093450, -0.902520365],
        ],
        dtype=np.float64,
    )
)

#: (hand model, side) -> the Quest -> hand fit (t, R). LH / RH: left / right hand; _V2: Wuji 2.
QUEST_TO_HAND = {
    ("wuji1", "left"): (QUEST_TO_LH_T, QUEST_TO_LH_R),
    ("wuji1", "right"): (QUEST_TO_RH_T, QUEST_TO_RH_R),
    ("wuji2", "left"): (QUEST_TO_LH_T_V2, QUEST_TO_LH_R_V2),
    ("wuji2", "right"): (QUEST_TO_RH_T_V2, QUEST_TO_RH_R_V2),
}


def _axis_rotation(axis: int, deg: float) -> np.ndarray:
    """Rotation by ``deg`` about coordinate axis ``axis`` (0 = x, 1 = y, 2 = z)."""
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    i, j = ((1, 2), (2, 0), (0, 1))[axis]
    R = np.eye(3)
    R[i, i] = R[j, j] = c
    R[i, j], R[j, i] = -s, s
    return R


#: Hand frame -> flange frame: Rz(90) @ Rx(-90), i.e. URDF rpy (-90, 0, 90) deg (fixed axes, x
#: first, then z; scipy ``from_euler("xyz", [-90, 0, 90], degrees=True)``): hand x -> flange +y,
#: hand y -> flange -z, hand z -> flange -x.
HAND_TO_FLANGE_R: np.ndarray = _axis_rotation(2, 90.0) @ _axis_rotation(0, -90.0)
#: Flange-to-palm standoff along flange z without the spacer.
HAND_TO_FLANGE_STANDOFF_M: float = 0.0179
#: The controller bracket's spacer between flange and hand, along flange z.
QUEST_BRACKET_SPACER_M: float = 0.012


def hand_to_flange_t(spacer: bool) -> np.ndarray:
    """Translation of T_hand_to_flange: the hand origin in flange coordinates, along flange z,
    without / with the spacer."""
    return np.array(
        [0.0, 0.0, HAND_TO_FLANGE_STANDOFF_M + (QUEST_BRACKET_SPACER_M if spacer else 0.0)],
        dtype=np.float64,
    )


def mount_transform(hand: str, side: str, spacer: bool = False) -> np.ndarray:
    """The mount ``M = T_hand_to_flange · T_quest_to_hand`` (the controller frame in flange
    coordinates) of one arm, with or without the spacer between flange and hand."""
    t, R = QUEST_TO_HAND[(hand, side)]
    return make_transform(hand_to_flange_t(spacer), HAND_TO_FLANGE_R) @ make_transform(t, R)


@dataclass
class HandConfig:
    """The ramp that opens the hand before and after an episode (and seats a start grasp)."""

    reset_update_rate: float = 100.0  # Hz
    reset_duration: float = 0.5  # s


@dataclass
class ControlConfig:
    """The deploy loop."""

    frequency: float = 30.0  # Hz: the rate the demonstrations were recorded at
    chunk_rows: int = 8  # rows of each 16-row chunk that vote
    warmup_inferences: int = 2
    diffusion_steps: int = 10  # DDIM steps
    max_steps: int = 1000  # per policy episode


@dataclass
class DeployConfig:
    tianji: TianjiConfig = field(default_factory=TianjiConfig)
    hand: HandConfig = field(default_factory=HandConfig)
    control: ControlConfig = field(default_factory=ControlConfig)
