"""SE(3) pose math shared by training, deploy and calibration: xyzw quaternions, rot6d, relative
poses, rotation distances and means. Needs only numpy (a few helpers import scipy).

Quaternions are ``[x, y, z, w]`` (scalar last). The 6-D rotation ("rot6d") is the first two rows
of the rotation matrix, row-major, as in UMI's ``mat_to_rot6d`` (Zhou et al., "On the Continuity of
Rotation Representations in Neural Networks", CVPR 2019). A pose relative to a base frame is
``T_base^-1 · T`` written as ``[pos(3) | rot6d(6)]``: the policy's arm actions and end-effector
observations. A transform named ``T_a_to_b`` (written a -> b) maps a-coordinates to b-coordinates,
i.e. it is the pose of frame a in frame b.
"""

import numpy as np


def quat_to_mat(quat: np.ndarray) -> np.ndarray:
    """(..., 4) xyzw quaternions -> (..., 3, 3) rotation matrices (normalized first, dtype kept)."""
    x, y, z, w = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    n = np.sqrt(x * x + y * y + z * z + w * w + 1e-12)
    x, y, z, w = x / n, y / n, z / n, w / n

    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    R = np.empty(quat.shape[:-1] + (3, 3), dtype=quat.dtype)
    R[..., 0, 0] = 1 - 2 * (yy + zz)
    R[..., 0, 1] = 2 * (xy - wz)
    R[..., 0, 2] = 2 * (xz + wy)
    R[..., 1, 0] = 2 * (xy + wz)
    R[..., 1, 1] = 1 - 2 * (xx + zz)
    R[..., 1, 2] = 2 * (yz - wx)
    R[..., 2, 0] = 2 * (xz - wy)
    R[..., 2, 1] = 2 * (yz + wx)
    R[..., 2, 2] = 1 - 2 * (xx + yy)
    return R


def pos_quat_to_mat(pos: np.ndarray, quat: np.ndarray) -> np.ndarray:
    """(..., 3) positions + (..., 4) xyzw quaternions -> (..., 4, 4) homogeneous transforms."""
    T = np.zeros(pos.shape[:-1] + (4, 4), dtype=pos.dtype)
    T[..., :3, :3] = quat_to_mat(quat)
    T[..., :3, 3] = pos
    T[..., 3, 3] = 1.0
    return T


def mat_to_pos_rot6d(mat: np.ndarray) -> np.ndarray:
    """(..., 4, 4) transforms -> (..., 9) ``[x, y, z, r00, r01, r02, r10, r11, r12]``."""
    rot6d = mat[..., :3, :3][..., :2, :].reshape(mat.shape[:-2] + (6,))
    return np.concatenate([mat[..., :3, 3], rot6d], axis=-1)


def rot6d_to_mat(rot6d: np.ndarray) -> np.ndarray:
    """(..., 6) rot6d -> (..., 3, 3) float64 rotations, by Gram-Schmidt on the two stored rows.

    The inverse of the rotation part of :func:`mat_to_pos_rot6d` (UMI's ``rot6d_to_mat``).
    """
    a1 = rot6d[..., :3].astype(np.float64)
    a2 = rot6d[..., 3:].astype(np.float64)
    b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + 1e-12)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = b2 / (np.linalg.norm(b2, axis=-1, keepdims=True) + 1e-12)
    b3 = np.cross(b1, b2, axis=-1)
    return np.stack([b1, b2, b3], axis=-2)


def make_transform(t, R) -> np.ndarray:
    """(4, 4) float64 rigid transform with rotation ``R`` (3, 3) and translation ``t`` (3,)."""
    T = np.eye(4)
    T[:3, :3] = np.asarray(R, dtype=np.float64).reshape(3, 3)
    T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T


def mat_to_pos_quat(T) -> tuple[np.ndarray, np.ndarray]:
    """(4, 4) rigid transform -> position (3,) and xyzw quaternion (4,), float64."""
    from scipy.spatial.transform import Rotation

    T = np.asarray(T, dtype=np.float64)
    return T[:3, 3].copy(), Rotation.from_matrix(T[:3, :3]).as_quat()


def rotation_angle_deg(Ra, Rb) -> float:
    """Geodesic angle between two rotation matrices, degrees: the norm of ``log(Ra^T Rb)``."""
    from scipy.spatial.transform import Rotation

    Ra, Rb = np.asarray(Ra, dtype=np.float64), np.asarray(Rb, dtype=np.float64)
    return float(np.rad2deg(np.linalg.norm(Rotation.from_matrix(Ra.T @ Rb).as_rotvec())))


def mean_quat(quats) -> np.ndarray:
    """Mean rotation of xyzw quaternions (N, 4): each is flipped into the hemisphere of the first
    (q and -q are one rotation), then they are averaged and normalized; the first quaternion if
    the mean vanishes."""
    q = np.asarray(quats, dtype=np.float64).reshape(-1, 4)
    q = q * np.where((q @ q[0]) < 0.0, -1.0, 1.0)[:, None]
    m = np.mean(q, axis=0)
    n = float(np.linalg.norm(m))
    return q[0].copy() if n < 1e-9 else m / n


def mean_transform(Ts) -> np.ndarray:
    """Mean of rigid transforms: the mean translation and the :func:`mean_quat` rotation."""
    from scipy.spatial.transform import Rotation

    Ts = [np.asarray(T, dtype=np.float64) for T in Ts]
    q = np.stack([Rotation.from_matrix(T[:3, :3]).as_quat() for T in Ts])
    t = np.mean([T[:3, 3] for T in Ts], axis=0)
    return make_transform(t, Rotation.from_quat(mean_quat(q)).as_matrix())


def invert_transform(mat: np.ndarray) -> np.ndarray:
    """Inverse of (..., 4, 4) rigid transforms, ``[R^T | -R^T t]`` (dtype kept)."""
    inv = np.zeros_like(mat)
    inv[..., :3, :3] = np.swapaxes(mat[..., :3, :3], -1, -2)
    inv[..., :3, 3] = -np.einsum("...ij,...j->...i", inv[..., :3, :3], mat[..., :3, 3])
    inv[..., 3, 3] = 1.0
    return inv


def compute_relative_transform(absolute_mats: np.ndarray, base_mat: np.ndarray) -> np.ndarray:
    """``base_mat^-1 · absolute_mats[t]`` for (T, 4, 4) transforms and one (4, 4) base."""
    return np.einsum("ij,tjk->tik", invert_transform(base_mat), absolute_mats)


def compute_relative_actions(ee_pos: np.ndarray, ee_quat: np.ndarray, base_idx: int) -> np.ndarray:
    """(T, 9) ``[pos | rot6d]`` of every frame's pose relative to the pose at ``base_idx``.

    ``ee_pos`` (T, 3) and ``ee_quat`` (T, 4, xyzw) are the absolute poses of one window.
    """
    abs_mats = pos_quat_to_mat(ee_pos, ee_quat)
    return mat_to_pos_rot6d(compute_relative_transform(abs_mats, abs_mats[base_idx]))


def relative_pose_9d(pos_a, quat_a, pos_b, quat_b) -> np.ndarray:
    """Per frame, pose b in the frame of pose a (``T_a^-1 · T_b``) as (..., 9) ``[pos | rot6d]``."""
    rel = np.einsum(
        "...ij,...jk->...ik",
        invert_transform(pos_quat_to_mat(pos_a, quat_a)),
        pos_quat_to_mat(pos_b, quat_b),
    )
    return mat_to_pos_rot6d(rel)
