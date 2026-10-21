"""Reward terms of the rotation skill; ``env_cfg.RewardsCfg`` weights them."""

from __future__ import annotations

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg


def _handle_velocity(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, state: str) -> torch.Tensor:
    """Handle angular velocity by finite differences over a control step.

    The previous angle is ``env.<state>`` and is not reset with an episode: the first call returns
    0, and the first step after a reset differences against the angle before the reset.
    """
    angle = env.scene[asset_cfg.name].data.joint_pos[:, asset_cfg.joint_ids].squeeze(-1)
    prev = getattr(env, state, angle)
    setattr(env, state, angle.clone())
    return (angle - prev) / env.step_dt


def handle_rotation(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, clip_min: float = -4.0, clip_max: float = 4.0
) -> torch.Tensor:
    """Handle angular velocity (finite differences with its own previous-angle state), clipped."""
    vel = _handle_velocity(env, asset_cfg, "_handle_rotation_prev_angle")
    return torch.clamp(vel, min=clip_min, max=clip_max)


def pose_deviation(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, group: str) -> torch.Tensor:
    """Squared deviation of the joints from the initial grasp, summed over the thumb joints
    (``group="thumb"``) or over the index to pinky joints (``group="index_to_pinky"``)."""
    asset = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    default_pos = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    diff_sq = (joint_pos - default_pos) ** 2
    cache_attr = f"_pose_deviation_mask_{group}"
    if not hasattr(env, cache_attr):
        mask = torch.zeros(len(asset_cfg.joint_ids), device=env.device)
        for i, jid in enumerate(asset_cfg.joint_ids):
            is_thumb = "finger1" in asset.joint_names[jid]
            if (group == "thumb" and is_thumb) or (group == "index_to_pinky" and not is_thumb):
                mask[i] = 1.0
        setattr(env, cache_attr, mask)
    return (diff_sq * getattr(env, cache_attr)).sum(dim=-1)


def mechanical_work(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """``(sum_j |tau_j| |qdot_j|)^2``."""
    asset = env.scene[asset_cfg.name]
    torques = asset.data.applied_torque[:, asset_cfg.joint_ids]
    vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    return (torch.abs(torques) * torch.abs(vel)).sum(dim=-1) ** 2


def action_smoothness(env: ManagerBasedRLEnv, std: float) -> torch.Tensor:
    """``exp(-mean_j (a_t - a_{t-1})^2 / std^2)`` of the unclipped policy actions (``a_{t-1} = 0``
    after a reset)."""
    a = env.action_manager.action
    prev = env.action_manager.prev_action
    action_rate_l2 = torch.square(a - prev).mean(dim=-1)
    return torch.exp(-action_rate_l2 / (std**2))


def fingertip_proximity(
    env: ManagerBasedRLEnv, hand_cfg: SceneEntityCfg, screw_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """``clip(1 - d / threshold, 0, 1)`` with d the mean horizontal distance of the three
    ``hand_cfg`` bodies (thumb, index and middle ``link4`` origins) from the handle axis, which does
    not depend on the grip height."""
    hand = env.scene[hand_cfg.name]
    screw = env.scene[screw_cfg.name]
    handle_xy = screw.data.body_pos_w[:, screw_cfg.body_ids[0], :2]
    # three per-fingertip distances summed left to right: the floating-point order of the trained
    # reward
    d0, d1, d2 = (
        torch.norm(hand.data.body_pos_w[:, i, :2] - handle_xy, dim=-1) for i in hand_cfg.body_ids
    )
    mean_dist = (d0 + d1 + d2) / 3.0
    return torch.clamp(1.0 - mean_dist / threshold, min=0.0, max=1.0)


def over_speed(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    """``max(theta_dot - threshold, 0)`` (finite differences with its own previous-angle state).

    The handle's joint velocity limit (6.28 rad/s) is below the 10 rad/s threshold, so the term does
    not fire; it is kept as part of the reward recipe.
    """
    vel = _handle_velocity(env, asset_cfg, "_over_speed_prev_angle")
    return torch.clamp(vel - threshold, min=0.0)
