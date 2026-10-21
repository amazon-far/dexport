"""Closed-loop runner of the in-hand rotation policy on the real hand.

Rebuilds the sim's observation and action terms (:mod:`dexport.skills.rotation.contract`):
:meth:`RotationRunner.start` resets the joint targets to the init grasp and primes the observation
history with the measured pose, and each :meth:`RotationRunner.step` observes the measured joints,
runs the actor and accumulates its clipped, masked action into the targets. Vectors are in
articulation order and the LEFT-hand convention; a caller driving the right hand mirrors readings
and targets with :func:`dexport.common.joints.mirror_abduction`.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from dexport.skills.rotation.contract import (
    ACTION_CLIP,
    ACTION_SCALE,
    HAND_ROOT_POS,
    HAND_ROOT_QUAT_WXYZ,
    HISTORY_LENGTH,
    NUM_JOINTS,
    OBS_CLIP,
    OBS_DIM,
    ROOT_POS_SLICE,
    ROOT_QUAT_SLICE,
    action_mask_art,
    init_grasp_art,
    joint_limits_art,
)


class RotationRunner:
    """Per-hand state of the rotation policy: the joint-target accumulator and the observation
    history. The actor (``obs_dim``, ``act_dim``, ``obs_norm_mean()``, ``act_np(obs)``) is
    read-only, so one actor can be shared by the runners of both hands."""

    def __init__(self, actor):
        if actor.obs_dim != OBS_DIM or actor.act_dim != NUM_JOINTS:
            raise ValueError(
                f"rotation actor has obs_dim={actor.obs_dim}, act_dim={actor.act_dim}; the "
                f"rotation skill uses a {OBS_DIM}-D observation and a {NUM_JOINTS}-D action"
            )
        self.actor = actor
        self.lower_art, self.upper_art = joint_limits_art()
        self.init_grasp_art = init_grasp_art()
        self.mask_art = action_mask_art()

        norm_mean = actor.obs_norm_mean()
        self.root_from_ckpt = norm_mean is not None
        if norm_mean is None:
            self.root_pos = np.array(HAND_ROOT_POS, dtype=np.float32)
            self.root_quat = np.array(HAND_ROOT_QUAT_WXYZ, dtype=np.float32)
        else:
            self.root_pos = norm_mean[ROOT_POS_SLICE].astype(np.float32)
            self.root_quat = norm_mean[ROOT_QUAT_SLICE].astype(np.float32)

        self.targets_art = self.init_grasp_art.copy()
        # The clipped, pre-mask action of the last step: what the policy observes next.
        self.prev_action = np.zeros(NUM_JOINTS, dtype=np.float32)
        self._hist_pos: deque = deque(maxlen=HISTORY_LENGTH)
        self._hist_tgt: deque = deque(maxlen=HISTORY_LENGTH)

    def start(self, meas_art: np.ndarray) -> None:
        """Begin an episode: targets back to the init grasp and no previous action. The
        joint-position history is filled with the measured pose (as the sim reset does) and the
        target history with the init grasp (the sim's starts from the PD target before the
        reset)."""
        self.targets_art = self.init_grasp_art.copy()
        self.prev_action = np.zeros(NUM_JOINTS, dtype=np.float32)
        self._hist_pos.clear()
        self._hist_tgt.clear()
        meas = np.asarray(meas_art, dtype=np.float64)
        for _ in range(HISTORY_LENGTH):
            self._hist_pos.append(meas.copy())
            self._hist_tgt.append(self.targets_art.copy())

    def step(self, meas_art: np.ndarray) -> np.ndarray:
        """One policy step from the measured joints; returns the new joint targets."""
        self._hist_pos.append(np.asarray(meas_art, dtype=np.float64).copy())
        self._hist_tgt.append(self.targets_art.copy())
        action = self.actor.act_np(self.build_obs())
        return self._apply_action(action)

    def build_obs(self) -> np.ndarray:
        """The actor observation from the current histories, as float32."""
        obs = np.concatenate(
            [
                np.concatenate(list(self._hist_pos), axis=0),
                np.concatenate(list(self._hist_tgt), axis=0),
                self.root_pos.astype(np.float64),
                self.root_quat.astype(np.float64),
                self.prev_action.astype(np.float64),
            ]
        )
        return np.clip(obs, -OBS_CLIP, OBS_CLIP).astype(np.float32)

    def _apply_action(self, raw_action: np.ndarray) -> np.ndarray:
        a = np.clip(np.asarray(raw_action, dtype=np.float64), -ACTION_CLIP, ACTION_CLIP)
        self.prev_action = a.astype(np.float32).copy()
        a = a * self.mask_art
        self.targets_art = np.clip(
            self.targets_art + ACTION_SCALE * a, self.lower_art, self.upper_art
        )
        return self.targets_art
