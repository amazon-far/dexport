"""Wuji 1 and Wuji 2 as parameters of one skill controller.

A hand pose is a ``(5, 4)`` float64 grid: rows thumb, index, middle, ring, pinky; columns the
finger's joints 1-4 (fingers 2-5: MCP flexion, abduction, PIP, DIP). Its row-major flatten is the
MuJoCo order of :mod:`dexport.common.joints`. Poses are authored in the LEFT-hand convention and
converted at the hardware boundary (:meth:`HandModel.to_side` / :meth:`HandModel.from_side`): Wuji 1
negates the finger 2-5 abduction joints for the right hand, Wuji 2 uses the same numbers. The power
grasp blends fingers 2-5 from the neutral pose (fraction 0) to the closed pose (fraction 1); the
pinch and two-finger grasps take over some of those rows (``GRASP_ROWS``) along the same blend.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dexport.common.joints import WUJI1_JOINT_LIMITS, WUJI1_THUMB_BASE_NEUTRAL, mirror_abduction

GRASP_ROWS = {"power": (1, 2, 3, 4), "pinch": (1,), "two_finger": (1, 2)}
GRASP_DESC = {"power": "fingers 2-5", "pinch": "index", "two_finger": "index + middle"}


@dataclass(frozen=True)
class PowerGrasp:
    """Closed power grasp (fraction 1) of fingers 2-5, LEFT convention: one MCP / PIP / DIP curl
    shared by the four fingers, plus each finger's abduction."""

    mcp: float
    pip: float
    dip: float
    abduction: tuple[float, float, float, float]

    def fingers(self) -> np.ndarray:
        """Rows 1-4 (index..pinky) of the closed pose, ``(4, 4)``."""
        return np.array(
            [[self.mcp, abd, self.pip, self.dip] for abd in self.abduction], dtype=np.float64
        )


@dataclass(frozen=True, eq=False)
class HandModel:
    """Everything the skill controller and the collector need to know about one hand model."""

    name: str
    transport: str  # driver selected by dexport.hardware.hands.make_hand: "usb" | "eth"
    thumb_base_neutral: float  # neutral angle of joint (0, 0); every other neutral angle is 0
    power_grasp: PowerGrasp
    grasp_cycle: tuple[str, ...]  # grasps the index trigger cycles through, power first
    thumb_jog_limits: dict  # {(0, 0): (lo, hi), (0, 1): (lo, hi)} for the two jogged joints
    mirror_right: bool  # right hand = mirror_abduction(left pose), else identical numbers
    has_rotation_skill: bool  # the in-hand rotation skill exists for this hand

    def __post_init__(self):
        if self.grasp_cycle[0] != "power" or not set(self.grasp_cycle) <= set(GRASP_ROWS):
            raise ValueError(f"bad grasp cycle {self.grasp_cycle}")

    @property
    def records_readback_age(self) -> bool:
        """Ethernet readback can go stale, so its per-tick age is recorded."""
        return self.transport == "eth"

    def neutral_pose(self) -> np.ndarray:
        pose = np.zeros((5, 4), dtype=np.float64)
        pose[0, 0] = self.thumb_base_neutral
        return pose

    def grasp_target(self) -> np.ndarray:
        """The closed power grasp; the thumb row stays at neutral (the thumb follows the jog)."""
        pose = self.neutral_pose()
        pose[1:] = self.power_grasp.fingers()
        return pose

    def power_blend(self, frac: float) -> np.ndarray:
        neutral = self.neutral_pose()
        return neutral + frac * (self.grasp_target() - neutral)

    def finger_target(
        self, power_frac: float, override: str = "", override_frac: float = 0.0
    ) -> np.ndarray:
        """Finger pose for a power fraction, with the rows of ``override`` (a grasp name, or ""
        for none) driven by ``override_frac`` instead."""
        target = self.power_blend(power_frac)
        if override:
            rows = list(GRASP_ROWS[override])
            target[rows] = self.power_blend(override_frac)[rows]
        return target

    def to_side(self, pose: np.ndarray, side: str) -> np.ndarray:
        """LEFT-convention pose -> ``side``'s hardware convention, as a float64 copy."""
        if side == "right" and self.mirror_right:
            return mirror_abduction(pose)
        return np.array(pose, dtype=np.float64, copy=True)

    def from_side(self, pose: np.ndarray, side: str) -> np.ndarray:
        """A reading in ``side``'s hardware convention -> the LEFT convention (the same map)."""
        return self.to_side(pose, side)

    def next_grasp(self, current: str) -> str:
        i = self.grasp_cycle.index(current)
        return self.grasp_cycle[(i + 1) % len(self.grasp_cycle)]


WUJI1 = HandModel(
    name="wuji1",
    transport="usb",
    thumb_base_neutral=WUJI1_THUMB_BASE_NEUTRAL,
    power_grasp=PowerGrasp(
        mcp=np.pi / 2,
        pip=np.pi / 4,
        dip=np.pi / 4,
        abduction=(-0.22, 0.0, 0.0, 0.05),
    ),
    grasp_cycle=("power", "pinch"),
    # the URDF limits of the two jogged thumb joints
    thumb_jog_limits={
        (0, 0): WUJI1_JOINT_LIMITS["left_finger1_joint1"],
        (0, 1): WUJI1_JOINT_LIMITS["left_finger1_joint2"],
    },
    mirror_right=True,
    has_rotation_skill=True,
)

WUJI2 = HandModel(
    name="wuji2",
    transport="eth",
    thumb_base_neutral=0.0,
    power_grasp=PowerGrasp(
        mcp=1.55,  # pi/2 would exceed the 1.570 rad mcp_flex limit
        pip=0.97,
        dip=1.20,
        abduction=(-0.22, -0.07, 0.0, 0.05),
    ),
    grasp_cycle=("power", "two_finger"),
    # URDF limits of the thumb cmc_flex and cmc_abd joints
    thumb_jog_limits={(0, 0): (-1.187, 1.291), (0, 1): (-1.484, 0.698)},
    mirror_right=False,
    # the rotation policy is trained for Wuji 1's kinematics and range of motion
    has_rotation_skill=False,
)

HAND_MODELS = {m.name: m for m in (WUJI1, WUJI2)}
