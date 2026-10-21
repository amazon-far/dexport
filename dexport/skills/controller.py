"""Quest-input skill controller for one hand.

Each tick, :meth:`SkillController.update` turns one Quest controller's index trigger and stick into
the hand's ``(5, 4)`` joint command. Thumb mode: stick x jogs the thumb flexion (joint ``(0, 0)``)
and stick y its abduction (``(0, 1)``). Finger mode: stick y opens or closes the selected grasp
incrementally (down closes) and the index trigger selects the next grasp. Screw mode (hands with the
rotation skill) runs the learned in-hand rotation policy closed loop. "Screw mode" is the operator
state, "rotation" the learned skill.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

import numpy as np

from dexport.common.joints import articulation_to_mujoco, mujoco_to_articulation
from dexport.skills.hand_models import GRASP_DESC, GRASP_ROWS, HandModel
from dexport.skills.rotation.contract import POLICY_HZ

RESET_SETTLE_S = 0.3  # neutral is held this long after the reset ramp, so the hand catches up
SCREW_EASE_S = 2.0  # ease into the rotation start grasp
FINGER_MODE_TRIGGER_THRESHOLD = 0.5  # trigger deflection that counts as a press


@dataclass(frozen=True)
class SkillConfig:
    """Operator-input tuning (times in s, release rates in grasp fraction per s)."""

    grasp_smoothing_tau: float = 0.25  # easing time constant while closing or holding a grasp
    grasp_release_tau: float = 0.06  # easing time constant at a fast release
    grasp_release_deadzone: float = 0.5  # release rate below which the closing tau is kept
    grasp_release_full_rate: float = 3.0  # release rate at which grasp_release_tau applies fully
    # Thumb jog and grasp: a full stick deflection sweeps a jogged joint's range, or the grasp
    # fraction from 0 to 1, in stick_sweep_time; deflections below stick_deadzone are ignored.
    stick_sweep_time: float = 1.0
    stick_deadzone: float = 0.15
    reset_duration: float = 0.5  # ramp time of a reset to neutral


class Mode(Enum):
    IDLE = auto()
    RESETTING = auto()
    SCREW_READY = auto()  # easing to, then holding, the rotation start grasp
    SCREW_RUNNING = auto()  # rotation policy running
    SCREW_FROZEN = auto()  # policy stopped, pose held


SCREW_MODES = (Mode.SCREW_READY, Mode.SCREW_RUNNING, Mode.SCREW_FROZEN)
# What the action button does in screw mode: start -> freeze -> back to the start grasp.
_SCREW_BUTTON_NEXT = {
    Mode.SCREW_READY: Mode.SCREW_RUNNING,
    Mode.SCREW_RUNNING: Mode.SCREW_FROZEN,
    Mode.SCREW_FROZEN: Mode.SCREW_READY,
}


def _grid_to_art(grid: np.ndarray) -> np.ndarray:
    return mujoco_to_articulation(np.asarray(grid, dtype=np.float64).reshape(20))


def _art_to_grid(vec_art: np.ndarray) -> np.ndarray:
    return articulation_to_mujoco(np.asarray(vec_art, dtype=np.float64)).reshape(5, 4)


class SkillController:
    """Skill state machine of one hand, driven by one Quest controller.

    ``action`` is the internal pose (LEFT convention); :meth:`update` returns it in the hand's
    hardware convention. ``measured`` is the latest readback (hardware convention), set by the
    collector before each update; the rotation policy observes it. ``rotation`` is a
    :class:`~dexport.skills.rotation.runner.RotationRunner`, or None to disable screw mode.

    Outside screw mode the action button toggles thumb / finger mode. A pinch or two-finger grasp
    takes over its rows from the fraction that drove them, so selecting it never moves the hand;
    moving the power grasp takes the rows back. Grasp targets are eased with a time constant that
    shortens when a grasp is released quickly. The grip enters screw mode and eases the hand to the
    rotation start grasp; the action button then starts the policy (closed loop on the measured
    joints at ``POLICY_HZ``), freezes it, and eases back to the start grasp; the grip leaves screw
    mode from the start grasp or a frozen pose. A reset ramps to the neutral pose and holds it for
    ``RESET_SETTLE_S``.
    """

    def __init__(
        self, model: HandModel, hz: float, cfg: SkillConfig, side: str = "left", rotation=None
    ):
        if side not in ("left", "right"):
            raise ValueError(f"side must be 'left' or 'right', got {side!r}")
        if rotation is not None and not model.has_rotation_skill:
            raise ValueError(f"{model.name} has no in-hand rotation skill")
        self.model = model
        self._hz = hz
        self._cfg = cfg
        self._side = side
        self._rotation = rotation
        # Stick x is negated on the left so that the same push moves either thumb outward.
        self._thumb_jog_x_sign = -1.0 if side == "left" else 1.0
        self.action = model.neutral_pose()
        self.measured = model.neutral_pose()
        self._mode = Mode.IDLE
        self._joystick_mode = "thumb"
        self._grasp = "power"  # the grasp the stick drives in finger mode
        self._finger_override = ""  # grasp owning its rows; "" = the power grasp drives all
        self._grasp_frac = dict.fromkeys(GRASP_ROWS, 0.0)
        self._prev_grasp_frac = 0.0
        self._trigger_was_pressed = False
        self._jog_targets = self._neutral_jog_targets()
        self._reset_start_time = 0.0
        self._reset_start_pos = model.neutral_pose()
        self._start_grasp = None if rotation is None else _art_to_grid(rotation.init_grasp_art)
        self._ease_from_time = 0.0
        self._ease_from_pos = model.neutral_pose()
        self._runner_started = False  # the runner was (re)started for the current policy run
        self._next_policy_time = 0.0

    @property
    def reset_done(self) -> bool:
        """The reset ramp and its settle hold are over (safe to start recording)."""
        return self._mode is not Mode.RESETTING

    @property
    def rotation_available(self) -> bool:
        return self._rotation is not None

    # Discrete inputs: rising edges, routed by the collector.

    def press_action_button(self, now: float) -> None:
        if self._mode is Mode.RESETTING:
            return
        if self._mode is Mode.IDLE:
            self._joystick_mode = "finger" if self._joystick_mode == "thumb" else "thumb"
            print(f"[{self._side} hand] {self._joystick_mode} mode")
            return
        nxt = _SCREW_BUTTON_NEXT[self._mode]
        if nxt is Mode.SCREW_RUNNING:  # start right away, even in the middle of the ease-in
            self._runner_started = False
            print(f"[{self._side} hand] rotation policy running ({POLICY_HZ:.0f} Hz)")
        elif nxt is Mode.SCREW_FROZEN:
            print(f"[{self._side} hand] rotation policy frozen (button: back to start, grip: exit)")
        else:
            self._ease_from_pos = self.action.copy()
            self._ease_from_time = now
            print(f"[{self._side} hand] easing back to the start grasp")
        self._mode = nxt

    def toggle_screw_mode(self, now: float) -> None:
        """Grip edge: enter screw mode, or leave it (through a reset) unless the policy runs."""
        if self._rotation is None or self._mode is Mode.RESETTING:
            return
        if self._mode is Mode.IDLE:
            self._mode = Mode.SCREW_READY
            self._ease_from_pos = self.action.copy()
            self._ease_from_time = now
            self._runner_started = False
            print(f"[{self._side} hand] screw mode: easing to the start grasp")
        elif self._mode is Mode.SCREW_RUNNING:
            print(f"[{self._side} hand] freeze the policy (action button) before leaving")
        else:
            print(f"[{self._side} hand] leaving screw mode")
            self.reset_to_neutral(now)

    def reset_to_neutral(self, now: float) -> None:
        """Ramp to the neutral pose over ``reset_duration``, clearing the mode, grasp and jog
        state."""
        self._reset_start_pos = self.action.copy()
        self._reset_start_time = now
        self._mode = Mode.RESETTING
        self._joystick_mode = "thumb"
        self._grasp = "power"
        self._finger_override = ""
        self._grasp_frac = dict.fromkeys(GRASP_ROWS, 0.0)
        self._prev_grasp_frac = 0.0
        self._jog_targets = self._neutral_jog_targets()

    def update(
        self, trigger: float, joystick_x: float, joystick_y: float, now: float
    ) -> np.ndarray:
        """Advance one tick; returns the ``(5, 4)`` command in the hand's hardware convention."""
        cfg = self._cfg
        deadzone = cfg.stick_deadzone
        sweep_time = cfg.stick_sweep_time
        rate = 0.0 if sweep_time <= 0.0 else 1.0 / sweep_time
        dt_input = 1.0 / self._hz

        if self._mode is Mode.RESETTING:
            elapsed = now - self._reset_start_time
            alpha = min(elapsed / cfg.reset_duration, 1.0)
            neutral = self.model.neutral_pose()
            self.action = self._reset_start_pos + alpha * (neutral - self._reset_start_pos)
            if elapsed >= cfg.reset_duration + RESET_SETTLE_S:
                self._mode = Mode.IDLE
            self._prev_grasp_frac = 0.0
            return self._output(self.action)

        if self._mode in SCREW_MODES:  # the stick and trigger are ignored in screw mode
            if self._mode is Mode.SCREW_READY:
                self._ease_to_start_grasp(now)
            elif self._mode is Mode.SCREW_RUNNING:
                self._run_policy(now)
            self._prev_grasp_frac = 0.0
            return self._output(self.action)

        if self._joystick_mode == "thumb":
            self._update_joint_jog(joystick_x, joystick_y, deadzone, rate, dt_input)
        else:
            # The trigger edge is only tracked in finger mode, so a trigger held while switching
            # from thumb mode selects the next grasp on the first finger-mode tick.
            trigger_pressed = trigger > FINGER_MODE_TRIGGER_THRESHOLD
            if trigger_pressed and not self._trigger_was_pressed:
                self._grasp = self.model.next_grasp(self._grasp)
                self._engage_grasp()
                print(f"[{self._side} hand] grasp: {self._grasp} ({GRASP_DESC[self._grasp]})")
            self._trigger_was_pressed = trigger_pressed
            self._update_grasp_frac(joystick_y, deadzone, rate, dt_input)

        grasp_frac = self._grasp_frac[self._grasp]
        self._ease_toward(self._finger_target(), grasp_frac)
        self._prev_grasp_frac = grasp_frac
        for idx, value in self._jog_targets.items():  # the thumb follows the jog targets
            self.action[idx] = value
        return self._output(self.action)

    def _output(self, action: np.ndarray) -> np.ndarray:
        return self.model.to_side(action, self._side)

    def _neutral_jog_targets(self) -> dict:
        neutral = self.model.neutral_pose()
        return {
            idx: float(np.clip(neutral[idx], lo, hi))
            for idx, (lo, hi) in self.model.thumb_jog_limits.items()
        }

    def _override_frac(self) -> float:
        return self._grasp_frac[self._finger_override] if self._finger_override else 0.0

    def _finger_target(self) -> np.ndarray:
        return self.model.finger_target(
            self._grasp_frac["power"], self._finger_override, self._override_frac()
        )

    def _engage_grasp(self) -> None:
        """A newly selected pinch / two-finger grasp takes over its rows at the fraction they were
        driven at. Selecting power keeps the current owner until the power grasp is moved."""
        grasp = self._grasp
        if grasp == "power" or self._finger_override == grasp:
            return
        self._grasp_frac[grasp] = self._grasp_frac[self._finger_override or "power"]
        self._finger_override = grasp

    def _update_joint_jog(
        self, joystick_x: float, joystick_y: float, deadzone: float, rate: float, dt_input: float
    ) -> None:
        """Thumb control: stick x jogs the thumb flexion (0, 0) and stick y its
        abduction (0, 1); a full deflection sweeps the joint's ``thumb_jog_limits`` range in
        ``stick_sweep_time``. Stick x is negated on the left hand, so the same push moves either
        thumb outward."""
        joystick_x = joystick_x * self._thumb_jog_x_sign
        for axis_val, idx in ((joystick_x, (0, 0)), (joystick_y, (0, 1))):
            v = float(axis_val)
            if abs(v) < deadzone:
                continue
            lo, hi = self.model.thumb_jog_limits[idx]
            self._jog_targets[idx] = float(
                np.clip(self._jog_targets[idx] + v * rate * dt_input * (hi - lo), lo, hi)
            )

    def _update_grasp_frac(
        self, joystick_y: float, deadzone: float, rate: float, dt_input: float
    ) -> None:
        v = float(joystick_y)
        if abs(v) < deadzone:
            return
        delta = -v * rate * dt_input  # stick down (negative y) closes
        grasp = self._grasp
        self._grasp_frac[grasp] = float(np.clip(self._grasp_frac[grasp] + delta, 0.0, 1.0))
        if grasp == "power":
            self._finger_override = ""

    def _ease_toward(self, target: np.ndarray, grasp_frac: float) -> None:
        cfg = self._cfg
        dt = 1.0 / self._hz
        tau_close, tau_open = cfg.grasp_smoothing_tau, cfg.grasp_release_tau
        dz, full = cfg.grasp_release_deadzone, cfg.grasp_release_full_rate
        release_rate = max(0.0, (self._prev_grasp_frac - grasp_frac) / dt)
        if release_rate <= dz:
            tau = tau_close
        else:
            blend = min(1.0, (release_rate - dz) / max(1e-6, full - dz))
            tau = tau_close + blend * (tau_open - tau_close)
        alpha = 1.0 if tau <= 0.0 else 1.0 - np.exp(-dt / tau)
        self.action = self.action + alpha * (target - self.action)

    def _ease_to_start_grasp(self, now: float) -> None:
        alpha = min((now - self._ease_from_time) / SCREW_EASE_S, 1.0)
        start = self._ease_from_pos
        self.action = start + alpha * (self._start_grasp - start)
        if alpha >= 1.0:
            self.action = self._start_grasp.copy()

    def _run_policy(self, now: float) -> None:
        """Run the rotation policy at POLICY_HZ against the wall clock; hold in between."""
        meas_art = _grid_to_art(self.model.from_side(self.measured, self._side))
        if not self._runner_started:
            self._rotation.start(meas_art)
            self.action = _art_to_grid(self._rotation.targets_art)
            self._runner_started = True
            self._next_policy_time = now + 1.0 / POLICY_HZ
            return
        if now >= self._next_policy_time:
            self.action = _art_to_grid(self._rotation.step(meas_art))
            period = 1.0 / POLICY_HZ
            self._next_policy_time += period
            if self._next_policy_time <= now:  # fell more than a period behind: re-anchor
                self._next_policy_time = now + period
