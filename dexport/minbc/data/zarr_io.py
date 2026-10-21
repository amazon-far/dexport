"""Reader for the recorded zarr trajectory stores.

A store (``trajectories.zarr``) holds one group per trajectory (``traj_N``). Per-frame arrays of a
trajectory: ``action`` / ``action_r`` (commanded left / right hand targets), ``hand_joints`` /
``hand_joints_r`` (measured hand joints, 5 fingers x 4), ``pose_l`` / ``pose_r`` (end-effector
position) and ``quat_l`` / ``quat_r`` (xyzw quaternion), and camera arrays (``left_rgb``,
``right_rgb``; (T, H, W, 3) uint8). The trajectory length is ``len(action)``; every other array
has as many frames. A recording session keeps its store at ``<session>/trajectories.zarr``, or is
the store itself (flat layout).
"""

import os

import natsort
import numpy as np
import zarr

from dexport.common.channels import CHANNELS
from dexport.common.pose import relative_pose_9d

#: Interchangeable camera names. ``base_rgb`` is the only (wrist) camera of a single-camera
#: recording, and the camera of a config saved without ``im_key``: the reader resolves ``base_rgb``
#: and ``left_rgb`` to each other, and deploy opens the camera on the policy's hand side for it
#: (:func:`dexport.deploy.camera.camera_key`).
CAMERA_ALIASES = {"base_rgb": "left_rgb", "left_rgb": "base_rgb"}
#: Directory name of the trajectory store inside a recording session.
STORE_DIR = "trajectories.zarr"


def session_store(path: str) -> str | None:
    """The trajectory store of a session directory (``path`` itself if flat), or None."""
    path = os.path.normpath(path)
    for store in (os.path.join(path, STORE_DIR), path):
        if os.path.isfile(os.path.join(store, ".zgroup")):
            return store
    return None


def session_dir(store: str) -> str:
    """The session directory of a trajectory store (the inverse of :func:`session_store`)."""
    store = os.path.normpath(store)
    return os.path.dirname(store) if os.path.basename(store) == STORE_DIR else store


def session_name(store: str) -> str:
    return os.path.basename(session_dir(store))


def trajectory_names(store: str) -> list[str]:
    """The ``traj_N`` groups of a recording session's store, in natural order."""
    group = zarr.open_group(store, mode="r")
    return natsort.natsorted(k for k in group.group_keys() if k.startswith("traj_"))


def list_trajectories(path: str) -> list[str]:
    """Paths of every member of a store in natural order (the training loader skips members it
    cannot read, with a warning), or ``[path]`` for a single trajectory."""
    if not os.path.isfile(os.path.join(path, ".zgroup")):
        raise FileNotFoundError(
            f"{path!r} is not a zarr group (expected a trajectories.zarr store)"
        )
    group = zarr.open_group(path, mode="r")
    if "action" in group:
        return [path]
    return [os.path.join(path, k) for k in natsort.natsorted(group.keys())]


def count_steps(traj: str) -> int:
    return int(zarr.open_group(traj, mode="r")["action"].shape[0])


def resolve_camera(group, name: str) -> str | None:
    """The array holding camera ``name`` in a trajectory group (alias-aware), or None."""
    if name in group:
        return name
    alias = CAMERA_ALIASES.get(name)
    return alias if alias is not None and alias in group else None


def read_trajectory(traj: str, data_cfg, load_images: bool = True) -> dict[str, np.ndarray]:
    """One trajectory as ``(T, ...)`` arrays.

    Keys: ``action`` (T, 20) float32, the commanded hand targets of ``data_cfg.hand_side``;
    ``hand_qpos`` (T, 20), ``ee_pos`` (T, 3), ``ee_quat`` (T, 4) float32 of the same side. When
    ``data_cfg.bimanual``, the unprimed keys are the left side, ``action`` is (T, 40) = [left |
    right], and ``hand_qpos_r``, ``ee_pos_r``, ``ee_quat_r`` and ``rel_ee`` (T, 9) = the right
    pose in the left frame as [pos | rot6d] are added. With ``load_images``, ``img`` is
    (T, n_cam, H, W, 3) uint8 in ``data_cfg.im_key`` order. Reads only ``data_cfg.bimanual``,
    ``data_cfg.hand_side`` and ``data_cfg.im_key``. A missing channel or camera raises KeyError;
    an array with another frame count than ``action``, or a wrong width, raises ValueError.
    """
    group = zarr.open_group(traj, mode="r")
    T = int(group["action"].shape[0])

    def frames(name):
        if name not in group:
            raise KeyError(
                f"trajectory {traj!r} has no {name!r} array (has {sorted(group.keys())})"
            )
        a = np.asarray(group[name][:])
        if a.shape[0] != T:
            raise ValueError(
                f"trajectory {traj!r}: {name!r} has {a.shape[0]} frames, 'action' has {T}"
            )
        return a

    def read(name, width):
        a = frames(name).reshape(T, -1)
        if a.shape[1] != width:
            raise ValueError(
                f"trajectory {traj!r}: {name!r} has {a.shape[1]} values per frame, not {width}"
            )
        return a.astype(np.float32)

    bimanual = bool(data_cfg.bimanual)
    sides = ("left", "right") if bimanual else (data_cfg.hand_side,)
    out = {}
    for side, suffix in zip(sides, ("", "_r")):
        ch = CHANNELS[side]
        out["action" + suffix] = read(ch.command, 20)
        out["hand_qpos" + suffix] = read(ch.measured, 20)
        out["ee_pos" + suffix] = read(ch.pos, 3)
        out["ee_quat" + suffix] = read(ch.quat, 4)
    if bimanual:
        out["action"] = np.concatenate([out["action"], out.pop("action_r")], axis=1)
        out["rel_ee"] = relative_pose_9d(
            out["ee_pos"], out["ee_quat"], out["ee_pos_r"], out["ee_quat_r"]
        ).astype(np.float32)
    if load_images:
        cams = []
        for name in data_cfg.im_key:
            src = resolve_camera(group, name)
            if src is None:
                raise KeyError(f"trajectory {traj!r} has no camera {name!r}")
            cams.append(frames(src))
        out["img"] = np.stack(cams, axis=1)
    return out


def _blosc_single_threaded():
    """One blosc decode thread per DataLoader worker process; the workers are the parallelism."""
    try:
        import numcodecs.blosc

        numcodecs.blosc.use_threads = False
    except ImportError:
        pass


class ZarrFrameReader:
    """Reads single frames of one trajectory's cameras on demand (``data.im_stream``).

    Cheap to construct in the parent process: the zarr group is opened lazily and re-opened
    whenever the process changes, so no handle is shared into forked DataLoader workers.
    """

    def __init__(self, traj: str):
        self.path = os.path.realpath(traj)
        self._group = None
        self._pid = None
        self._cams = {}

    def _open(self):
        if self._group is None or self._pid != os.getpid():
            _blosc_single_threaded()
            self._group = zarr.open_group(self.path, mode="r")
            self._pid = os.getpid()
            self._cams = {}
        return self._group

    def resolve_cam(self, name: str) -> str | None:
        if name not in self._cams:
            self._cams[name] = resolve_camera(self._open(), name)
        return self._cams[name]

    def check_cameras(self, im_key, T: int, height: int, width: int) -> None:
        """Raise unless every ``im_key`` camera is present with (T, height, width, 3) frames."""
        for name in im_key:
            src = self.resolve_cam(name)
            if src is None:
                raise KeyError(f"trajectory {self.path!r} has no camera {name!r}")
            shape = tuple(self._open()[src].shape)
            if shape != (T, height, width, 3):
                raise ValueError(
                    f"trajectory {self.path!r}: camera {name!r} has shape {shape}, expected "
                    f"{(T, height, width, 3)}"
                )

    def get(self, frame_idx: int, cam_name: str) -> np.ndarray:
        """Frame ``frame_idx`` of camera ``cam_name`` as (H, W, 3) uint8."""
        src = self.resolve_cam(cam_name)
        if src is None:
            raise KeyError(f"trajectory {self.path!r} has no camera {cam_name!r}")
        return np.asarray(self._open()[src][frame_idx])
