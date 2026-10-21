"""Termination of the rotation skill when a fingertip leaves the handle."""

from __future__ import annotations

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg


def fingertip_too_far(
    env: ManagerBasedRLEnv, hand_cfg: SceneEntityCfg, screw_cfg: SceneEntityCfg, max_dist: float
) -> torch.Tensor:
    """True where any ``hand_cfg`` body (a fingertip ``link4`` origin) is more than ``max_dist``
    (3-D) from the handle body."""
    hand = env.scene[hand_cfg.name]
    screw = env.scene[screw_cfg.name]
    ft_pos = hand.data.body_pos_w[:, hand_cfg.body_ids, :3]
    handle_pos = screw.data.body_pos_w[:, screw_cfg.body_ids[0], :3].unsqueeze(1)
    dists = torch.norm(ft_pos - handle_pos, dim=-1)
    return (dists > max_dist).any(dim=-1)
