"""Event terms of the rotation env: the hand self-collision filter and the handle-joint friction
(startup), the random disturbance force on the handle (every step) and the jitter of the initial
grasp (reset)."""

from __future__ import annotations

import torch
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg


def filter_hand_self_collisions(
    env: ManagerBasedEnv, env_ids, asset_cfg: SceneEntityCfg, pairs: tuple[tuple[str, str], ...]
) -> None:
    """Startup: disable the collisions between the given link pairs of the ``asset_cfg``
    articulation in every env (UsdPhysics FilteredPairsAPI)."""
    import isaacsim.core.utils.prims as prim_utils
    from pxr import Sdf, UsdPhysics

    prim_name = env.scene[asset_cfg.name].cfg.prim_path.rsplit("/", 1)[-1]
    for env_path in env.scene.env_prim_paths:
        root = f"{env_path}/{prim_name}"
        for a, b in pairs:
            prim_a = prim_utils.get_prim_at_path(f"{root}/{a}")
            if not prim_a.IsValid():
                raise RuntimeError(f"filter_hand_self_collisions: prim not found: {root}/{a}")
            api = UsdPhysics.FilteredPairsAPI.Apply(prim_a)
            api.CreateFilteredPairsRel().AddTarget(Sdf.Path(f"{root}/{b}"))


def set_handle_joint_friction(
    env: ManagerBasedEnv, env_ids, asset_cfg: SceneEntityCfg, friction: float
) -> None:
    """Startup: write the friction (static = dynamic) of the ``asset_cfg`` joints into PhysX and
    check the readback.

    The URDF's ``<dynamics friction>`` does not reach the joint friction PhysX applies, so the
    resistance the fingers turn the handle against is set here.
    """
    asset = env.scene[asset_cfg.name]
    asset.write_joint_friction_coefficient_to_sim(
        friction, joint_dynamic_friction_coeff=friction, joint_ids=asset_cfg.joint_ids
    )
    got = asset.data.joint_friction_coeff[0, asset_cfg.joint_ids]
    if (got - friction).abs().max().item() > 1e-4:
        raise RuntimeError(f"wrote joint friction {friction}, but PhysX reads {got.tolist()}")


def apply_handle_disturbance(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    force_scale: float,
    prob: float,
    decay: float,
    asset_cfg: SceneEntityCfg,
) -> None:
    """Interval event (every control step): a random external force on the handle, in its body
    frame.

    Per env, with probability ``prob`` the force is redrawn as ``U(-1, 1)^3 * force_scale * m``,
    with ``m`` the handle's nominal URDF mass (``data.default_mass``: the mass randomization writes
    only to PhysX, so every env uses 0.056 kg); otherwise it decays by ``decay``. The force buffer
    lives on the env and carries over episode resets.
    """
    asset = env.scene[asset_cfg.name]
    device = asset.device
    n = len(env_ids)
    body_ids = asset_cfg.body_ids
    num_bodies = len(body_ids)

    force_buf = getattr(env, "_handle_disturbance_force", None)
    if force_buf is None:
        force_buf = torch.zeros(asset.num_instances, num_bodies, 3, device=device)
        env._handle_disturbance_force = force_buf

    default_mass = asset.data.default_mass
    mass = default_mass[env_ids.to(default_mass.device)][:, body_ids].to(device)  # (n, num_bodies)

    resample = torch.rand(n, device=device) < prob  # (n,)
    rand_dir = torch.empty(n, num_bodies, 3, device=device).uniform_(-1.0, 1.0)
    new_force = rand_dir * (force_scale * mass).unsqueeze(-1)  # (n, num_bodies, 3)

    force = torch.where(resample.view(n, 1, 1), new_force, force_buf[env_ids] * decay)
    force_buf[env_ids] = force

    asset.set_external_force_and_torque(
        forces=force, torques=torch.zeros_like(force), body_ids=body_ids, env_ids=env_ids
    )


def reset_joints_by_range_fraction(
    env: ManagerBasedEnv, env_ids: torch.Tensor, frac: float, asset_cfg: SceneEntityCfg
) -> None:
    """Reset: joint positions = initial grasp + ``U(-frac, frac) * (upper - lower)`` per env and
    joint (the ``asset_cfg`` joints), clamped to the limits; joint velocities zero."""
    asset = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids
    rows = env_ids[:, None]

    default_pos = asset.data.default_joint_pos[rows, joint_ids].clone()
    default_vel = asset.data.default_joint_vel[rows, joint_ids].clone()

    limits = asset.data.soft_joint_pos_limits[rows, joint_ids]  # (n, joints, 2)
    span = (limits[..., 1] - limits[..., 0]) * frac
    offset = (torch.rand_like(default_pos) * 2.0 - 1.0) * span
    joint_pos = (default_pos + offset).clamp_(limits[..., 0], limits[..., 1])

    asset.write_joint_state_to_sim(joint_pos, default_vel, joint_ids=joint_ids, env_ids=env_ids)
