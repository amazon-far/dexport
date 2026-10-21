"""Ground-truth replay: a recorded trajectory played through the deploy loop as the policy.

For the launch tick ``p``, :class:`RecordedChunks` returns the chunk the training dataset labels at
frame ``p``: arm rows are frames ``p+1 .. p+16`` relative to frame ``p`` (``[pos | rot6d]``), hand
rows are the recorded hand targets of those frames. Through the same HATO loop and mount as a
policy, the arm then commands the rigid image of the recorded controller motion, anchored at the
home flange pose — the calibration check before deploying a policy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dexport.common.channels import CHANNELS
from dexport.common.pose import compute_relative_actions
from dexport.deploy.errors import DeployError
from dexport.hardware.tianji.arms import ARM_SIDE
from dexport.minbc.config import DPConfig

#: Rows of a recorded chunk: the training chunk length.
CHUNK_LEN = DPConfig().total_horizon


@dataclass
class Recording:
    """Per arm: controller positions (T, 3), xyzw quaternions (T, 4), hand targets (T, 20)."""

    pos: dict
    quat: dict
    hand: dict
    n_frames: int


def load_recording(zarr_path: str, traj: str, arms) -> Recording:
    """Read one trajectory's channels for ``arms`` (float64, not cast). Refuses a recording whose
    channels have different lengths or contain non-finite values (the collector NaN-fills a
    controller it never saw), naming the channel; two arms are cut to the shorter one."""
    import zarr

    store = zarr.open_group(str(zarr_path), mode="r")
    if traj not in store:
        raise DeployError(f"{traj!r} not in {zarr_path}. Trajectories: {sorted(store.keys())[:20]}")
    grp = store[traj]
    pos, quat, hand, lens = {}, {}, {}, {}
    for a in arms:
        side = ARM_SIDE[a]
        ch = CHANNELS[side]
        action_k, joints_k, pos_k, quat_k = ch.command, ch.measured, ch.pos, ch.quat
        for k in (pos_k, quat_k, action_k):
            if k not in grp:
                raise DeployError(
                    f"trajectory {traj!r} (arm {a}, {side} hand) has no {k!r} channel; it has "
                    f"{sorted(grp.keys())} (a right-hand recording needs --arm B, a left-hand one "
                    "--arm A)"
                )
        ch = {k: np.asarray(grp[k], dtype=np.float64) for k in (pos_k, quat_k, action_k)}
        if joints_k in grp:  # not replayed, but a broken measured channel means a broken recording
            ch[joints_k] = np.asarray(grp[joints_k], dtype=np.float64)
        n = {k: int(v.shape[0]) for k, v in ch.items()}
        if len(set(n.values())) != 1:
            raise DeployError(
                f"trajectory {traj!r} has ragged channels {n}: every channel needs one row per "
                "recorded tick"
            )
        for k, v in ch.items():
            bad = ~np.isfinite(v)
            if bad.any():
                rows = np.unique(np.nonzero(bad)[0])
                raise DeployError(
                    f"trajectory {traj!r} channel {k!r} has {int(bad.sum())} non-finite values in "
                    f"{rows.size} rows (first {rows[:5].tolist()}): the collector NaN-fills a "
                    f"controller it never saw, so this recording has no valid {side} data"
                )
        pos[a], quat[a] = ch[pos_k], ch[quat_k]
        hand[a] = ch[action_k].reshape(len(ch[action_k]), -1)
        lens[a] = n[pos_k]
    n_frames = min(lens.values())
    if n_frames < 2:
        raise DeployError(f"trajectory {traj!r} has {n_frames} frame(s); a replay needs at least 2")
    return Recording(
        pos={a: pos[a][:n_frames] for a in arms},
        quat={a: quat[a][:n_frames] for a in arms},
        hand={a: hand[a][:n_frames] for a in arms},
        n_frames=n_frames,
    )


class RecordedChunks:
    """``predict`` for the deploy loop: the recording's chunk for each launch tick (all arms'
    9-D rows first, then all hands' 20-D rows). Frames past the end repeat the last one."""

    def __init__(self, rec: Recording, arms):
        self.rec, self.arms = rec, tuple(arms)

    def __call__(self, req) -> np.ndarray:
        p = int(req.launch_step)
        idx = np.minimum(np.r_[p, p + 1 + np.arange(CHUNK_LEN)], self.rec.n_frames - 1)
        blocks = [
            compute_relative_actions(self.rec.pos[a][idx], self.rec.quat[a][idx], base_idx=0)[1:]
            for a in self.arms
        ]
        blocks += [self.rec.hand[a][idx[1:]] for a in self.arms]
        return np.concatenate(blocks, axis=1)
