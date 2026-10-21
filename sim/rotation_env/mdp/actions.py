"""Finger action of the rotation skill: delta joint targets.

Every control step ``target <- clamp(target + ACTION_SCALE * mask * clip(a, -1, 1), lower, upper)``;
the mask zeroes ring and pinky, so 12 joints are controlled, and a reset returns the target to the
initial grasp. Only this update clips the action: the previous-action observation and the
smoothness reward read the unclipped policy output (``ActionManager.action``). The hardware runner
observes the clipped action, which is the same whenever the policy mean stays within +-1.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING

import torch
from isaaclab.assets.articulation import Articulation
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from dexport.skills.rotation.contract import ACTION_CLIP, ACTION_SCALE, FIXED_FINGERS


class DeltaJointTargetAction(ActionTerm):
    """Accumulates clipped, masked actions into persistent joint position targets."""

    cfg: DeltaJointTargetActionCfg
    _asset: Articulation

    def __init__(self, cfg: DeltaJointTargetActionCfg, env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)
        self._joint_ids, self._joint_names = self._asset.find_joints(cfg.joint_names)
        self._num_joints = len(self._joint_ids)
        if self._num_joints == self._asset.num_joints:
            self._joint_ids = slice(None)
        self._scale = cfg.scale
        self._clip = cfg.clip
        self._clipped_actions = torch.zeros(self.num_envs, self._num_joints, device=self.device)
        self._targets = self._asset.data.default_joint_pos[:, self._joint_ids].clone()
        # Views of the articulation's soft limits (all joints are selected, so the index is a
        # slice): the policy joint limits written after env creation reach the clamp through them.
        self._lower = self._asset.data.soft_joint_pos_limits[:, self._joint_ids, 0]
        self._upper = self._asset.data.soft_joint_pos_limits[:, self._joint_ids, 1]
        mask = torch.ones(self._num_joints, device=self.device)
        for i, name in enumerate(self._joint_names):
            if any(finger in name for finger in cfg.masked_fingers):
                mask[i] = 0.0
        self._action_mask = mask

    @property
    def action_dim(self) -> int:
        return self._num_joints

    @property
    def raw_actions(self) -> torch.Tensor:
        """Isaac Lab's ``raw_actions``: here the clipped, unmasked action. The previous-action
        observation calls ``last_action()`` without ``action_name``, i.e. ``ActionManager.action``,
        the unclipped policy output; naming this term there would change the trained observation."""
        return self._clipped_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._targets

    def process_actions(self, actions: torch.Tensor) -> None:
        actions = torch.clamp(actions, -self._clip, self._clip)
        self._clipped_actions[:] = actions
        actions = actions * self._action_mask
        self._targets = torch.clamp(self._targets + self._scale * actions, self._lower, self._upper)

    def apply_actions(self) -> None:
        self._asset.set_joint_position_target(self._targets, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._clipped_actions[env_ids] = 0.0
        self._targets[env_ids] = self._asset.data.default_joint_pos[env_ids][:, self._joint_ids]


@configclass
class DeltaJointTargetActionCfg(ActionTermCfg):
    """Configuration of :class:`DeltaJointTargetAction`."""

    class_type: type = DeltaJointTargetAction
    joint_names: list[str] = MISSING
    """Joint-name expressions of the controlled articulation joints."""
    scale: float = ACTION_SCALE
    """Joint-target change per unit action and control step (rad)."""
    masked_fingers: tuple[str, ...] = FIXED_FINGERS
    """Finger-name substrings whose joints get no action (``finger4``, ``finger5``: ring, pinky)."""
    clip: float = ACTION_CLIP
    """Symmetric bound on the policy action before it is accumulated."""
