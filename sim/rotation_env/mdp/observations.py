"""Observation terms of the rotation env that Isaac Lab does not provide."""

from __future__ import annotations

import torch
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg


def joint_pos_target(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Joint position targets of the articulation's PD drive.

    This is the buffer the action writes each step, so the first observation after env creation is
    all zeros and the observation returned on a reset step shows the target from before the reset.
    """
    asset = env.scene[asset_cfg.name]
    return asset.data.joint_pos_target[:, asset_cfg.joint_ids]


def fingertip_pos(env: ManagerBasedEnv, hand_cfg: SceneEntityCfg) -> torch.Tensor:
    """Positions of the ``hand_cfg`` bodies (the fingertip ``link4`` origins) relative to the env
    origin, flattened."""
    hand = env.scene[hand_cfg.name]
    pos = hand.data.body_pos_w[:, hand_cfg.body_ids, :3] - env.scene.env_origins.unsqueeze(1)
    return pos.reshape(pos.shape[0], -1)


def handle_pos(env: ManagerBasedEnv, screw_cfg: SceneEntityCfg) -> torch.Tensor:
    """Position of the handle body relative to the env origin."""
    screw = env.scene[screw_cfg.name]
    return screw.data.body_pos_w[:, screw_cfg.body_ids[0], :3] - env.scene.env_origins
