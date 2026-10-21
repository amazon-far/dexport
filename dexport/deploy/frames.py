"""Mount frame math between the policy (Quest-controller frame) and the arm (flange frame).

The policy was trained on Quest-controller poses; the robot measures and commands flange poses.
With the controller frame rigidly attached to the flange by the mount ``M`` (the controller frame
in flange coordinates, :func:`dexport.deploy.config.mount_transform`), a relative arm action maps as
``A_flange = M A_quest M^-1`` (:func:`conjugate_actions`) and an end-effector
observation the inverse way. Poses are ``[pos(3) | rot6d(6)]`` relative poses
(:mod:`dexport.common.pose`); quaternions are xyzw.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from dexport.common.pose import make_transform, mat_to_pos_rot6d, rot6d_to_mat
from dexport.minbc.config import ARM_ACTION_DIM, HAND_ACTION_DIM


def action_slices(n_arms: int) -> tuple[tuple[slice, ...], tuple[slice, ...]]:
    """Columns of each arm's 9-D relative pose and of each hand's 20-D target in the policy
    action: all arms first, then all hands (29-D for one arm, 58-D for two)."""
    arms = tuple(slice(ARM_ACTION_DIM * i, ARM_ACTION_DIM * (i + 1)) for i in range(n_arms))
    h0 = ARM_ACTION_DIM * n_arms
    hands = tuple(
        slice(h0 + HAND_ACTION_DIM * i, h0 + HAND_ACTION_DIM * (i + 1)) for i in range(n_arms)
    )
    return arms, hands


def conjugate_actions(rel9, M: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Relative actions ``[pos | rot6d]`` (N, 9) from the controller frame into the flange frame,
    ``M A M^-1``: rotations ``M_R R M_R^-1`` (N, 3, 3) and translations
    ``M_R p + M_t - R_flange M_t`` (N, 3), in which the mount's lever arm couples every rotation
    into a translation."""
    rel = np.asarray(rel9, dtype=np.float64).reshape(-1, 9)
    R = rot6d_to_mat(rel[:, 3:9])
    M_R, M_t = M[:3, :3], M[:3, 3]
    R_ee = np.einsum("ij,njk,kl->nil", M_R, R, np.linalg.inv(M_R))
    p_ee = (rel[:, :3] @ M_R.T) + M_t - (R_ee @ M_t)
    return R_ee, p_ee


def chunk_to_flange_targets(rel9, base_pos, base_quat, M) -> tuple[np.ndarray, np.ndarray]:
    """Absolute flange targets of a chunk of relative arm actions (N, 9): each row conjugated by the
    mount (:func:`conjugate_actions`) and composed onto the chunk's base flange pose
    ``(base_pos, base_quat)``. Returns ``(N, 3)`` positions and ``(N, 4)`` xyzw quaternions in
    float32, the precision of the policy's actions."""
    R_ee, p_ee = conjugate_actions(rel9, M)
    R_base = Rotation.from_quat(base_quat).as_matrix()
    pos = (base_pos + p_ee @ R_base.T).astype(np.float32)
    quat = Rotation.from_matrix(np.einsum("ij,njk->nik", R_base, R_ee)).as_quat()
    return pos, quat.astype(np.float32)


def remap_ee_obs_window(window: list, M: np.ndarray, suffix: str = "") -> list:
    """In place: each step's relative end-effector pose (``ee_pos{suffix}``, ``ee_quat{suffix}``,
    measured in the flange frame) into the controller frame the policy was trained in,
    ``M^-1 D M`` (float32)."""
    M_inv = np.linalg.inv(M)
    pk, qk = f"ee_pos{suffix}", f"ee_quat{suffix}"
    for step in window:
        p = np.asarray(step[pk], dtype=np.float64)
        r6 = np.asarray(step[qk], dtype=np.float64)
        p9 = mat_to_pos_rot6d(M_inv @ make_transform(p, rot6d_to_mat(r6)) @ M)
        step[pk] = p9[:3].astype(np.float32)
        step[qk] = p9[3:].astype(np.float32)
    return window


def remap_rel_ee_window(window: list, M_left: np.ndarray, M_right: np.ndarray) -> list:
    """In place: the right hand's pose relative to the left (``rel_ee``, from the two measured
    flange poses) into the controller frames, ``M_left^-1 A M_right``. Steps without ``rel_ee``
    are left as they are."""
    M_left_inv = np.linalg.inv(M_left)
    for step in window:
        if "rel_ee" not in step:
            continue
        rel = np.asarray(step["rel_ee"], dtype=np.float64)
        A = make_transform(rel[:3], rot6d_to_mat(rel[3:9]))
        step["rel_ee"] = mat_to_pos_rot6d(M_left_inv @ A @ M_right).astype(np.float32)
    return window


def to_policy_frame(window: list, mounts: list) -> list:
    """In place: an observation window built from flange poses, in the policy's frame. One mount
    per arm, in arm order (the second arm's keys carry the ``_r`` suffix)."""
    remap_ee_obs_window(window, mounts[0])
    if len(mounts) == 2:
        remap_ee_obs_window(window, mounts[1], "_r")
        remap_rel_ee_window(window, mounts[0], mounts[1])
    return window
