"""Construction of the in-hand rotation environment.

:func:`build_env_cfg` returns the training configuration; ``randomize=False`` gives the nominal
plant without domain randomization. :func:`make_env` creates the ``ManagerBasedRLEnv``, checks it
against the interface the hardware runner relies on (:mod:`dexport.skills.rotation.contract`) and
writes the policy joint limits. Import this module after ``AppLauncher`` has started Isaac Sim.
"""

from __future__ import annotations

import math

from isaaclab.envs import ManagerBasedRLEnv

from dexport.common.joints import ARTICULATION_ORDER
from dexport.skills.rotation.contract import (
    CRITIC_OBS_DIM,
    NUM_JOINTS,
    OBS_DIM,
    POLICY_HZ,
    POLICY_JOINT_LIMITS,
    ROOT_POS_SLICE,
    ROOT_QUAT_SLICE,
)
from rotation_env.env_cfg import RANDOMIZATION_EVENTS, RotationEnvCfg
from rotation_env.ppo import NUM_ENVS, SEED


def build_env_cfg(
    num_envs: int = NUM_ENVS, seed: int = SEED, device: str = "cuda:0", randomize: bool = True
) -> RotationEnvCfg:
    """The training environment configuration of the rotation skill."""
    cfg = RotationEnvCfg()
    cfg.scene.num_envs = num_envs
    cfg.seed = seed
    cfg.sim.device = device
    if not randomize:
        missing = set(RANDOMIZATION_EVENTS) - set(vars(cfg.events))
        if missing:
            raise KeyError(f"RANDOMIZATION_EVENTS without an EventsCfg term: {sorted(missing)}")
        for name in RANDOMIZATION_EVENTS:
            setattr(cfg.events, name, None)
    return cfg


def make_env(cfg: RotationEnvCfg, render_mode: str | None = None) -> ManagerBasedRLEnv:
    """Create the environment, check its interface and apply the policy joint limits."""
    env = ManagerBasedRLEnv(cfg=cfg, render_mode=render_mode)
    try:
        _check_interface(env)
        _apply_policy_joint_limits(env.scene["hand"])
    except BaseException:
        env.close()  # Isaac Sim cannot shut down while an environment is open
        raise
    return env


def _check_interface(env: ManagerBasedRLEnv) -> None:
    """Raise unless the env matches the observation/action layout the hardware runner uses."""
    hand = env.scene["hand"]
    if list(hand.joint_names) != ARTICULATION_ORDER:
        raise RuntimeError(f"hand joint order {hand.joint_names} != {ARTICULATION_ORDER}")
    obs = env.observation_manager
    dims = {g: tuple(obs.group_obs_dim[g]) for g in ("policy", "critic")}
    if dims != {"policy": (OBS_DIM,), "critic": (CRITIC_OBS_DIM,)}:
        raise RuntimeError(
            f"observation dims {dims} != {OBS_DIM} (policy), {CRITIC_OBS_DIM} (critic)"
        )
    offsets, start = {}, 0
    for name, dim in zip(obs.active_terms["policy"], obs.group_obs_term_dim["policy"]):
        offsets[name] = slice(start, start + dim[0])
        start += dim[0]
    root = (offsets.get("hand_root_pos"), offsets.get("hand_root_quat"))
    if root != (ROOT_POS_SLICE, ROOT_QUAT_SLICE):
        raise RuntimeError(f"policy observation root pose at {root}, expected the contract slices")
    if env.action_manager.total_action_dim != NUM_JOINTS:
        raise RuntimeError(f"action dim {env.action_manager.total_action_dim} != {NUM_JOINTS}")
    if not math.isclose(env.step_dt, 1.0 / POLICY_HZ):
        raise RuntimeError(f"control period {env.step_dt} s != 1/{POLICY_HZ} s")


def _apply_policy_joint_limits(hand) -> None:
    """Tighten the hand joint limits to ``POLICY_JOINT_LIMITS`` (None keeps the URDF bound).

    Written after env creation; the finger action clamps against these limits because its bounds
    are views of the articulation's soft-limit buffer, which this write updates in place.
    """
    limits = hand.data.joint_pos_limits
    for name, (lower, upper) in POLICY_JOINT_LIMITS.items():
        joint = hand.find_joints(name, preserve_order=True)[0][0]
        new = limits[:, joint, :].clone()
        if lower is not None:
            new[:, 0] = lower
        if upper is not None:
            new[:, 1] = upper
        hand.write_joint_position_limit_to_sim(
            new.unsqueeze(1), joint_ids=[joint], warn_limit_violation=False
        )
