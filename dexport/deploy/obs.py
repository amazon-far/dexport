"""The policy's observation window, built from the deploy loop's observation history.

Reproduces the training dataset's window (``dexport.minbc.data.dataset``): ``obs_horizon`` steps,
oldest first, each with the measured hand state and the end-effector pose relative to the current
frame (the newest history entry). ``include_current`` is the checkpoint's ``dp.obs_current_step``:
True = the last ``obs_horizon`` frames including the current one (its relative pose is the
identity), False = the ``obs_horizon`` frames before it. A short history is left-padded by
repeating its oldest entry. The poses are in the flange frame;
:func:`~dexport.deploy.frames.to_policy_frame` maps the window into the policy's frame.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from dexport.common.pose import compute_relative_actions, relative_pose_9d
from dexport.minbc.config import WINDOW_IMAGES_KEY


def new_history(arms, maxlen: int, with_images: bool) -> dict:
    """An empty observation history, as :func:`obs_window` reads it: per arm, deques of the
    measured hand state (``hand``), flange position (``ee_pos``) and xyzw quaternion
    (``ee_quat``), plus the camera stacks (``img``, None without cameras)."""
    return {
        "hand": {a: deque(maxlen=maxlen) for a in arms},
        "ee_pos": {a: deque(maxlen=maxlen) for a in arms},
        "ee_quat": {a: deque(maxlen=maxlen) for a in arms},
        "img": deque(maxlen=maxlen) if with_images else None,
    }


def _tail(entries, n: int) -> list:
    """The last ``n`` entries, left-padded by repeating the oldest one."""
    seq = list(entries)
    if not seq:
        raise RuntimeError("the observation history is empty")
    return ([seq[0]] * max(0, n - len(seq)) + seq)[-n:]


def obs_window(hist: dict, arms, obs_horizon: int, include_current: bool, rel_ee=False) -> list:
    """Per-step observation dicts for the policy.

    ``hist`` is a history as :func:`new_history` makes it, oldest first (``img`` entries are
    ``(n_cam, H, W, 3)`` uint8 stacks; a missing ``img`` means no cameras). The first arm's keys
    are ``hand_qpos``, ``ee_pos``, ``ee_quat``; the second arm's carry the ``_r`` suffix, and
    ``rel_ee`` adds the second arm's pose relative to the first. The camera stack goes under
    ``WINDOW_IMAGES_KEY``.
    """
    H = int(obs_horizon)
    T = H if include_current else H + 1
    cols, poses = {}, []
    for i, arm in enumerate(arms):
        sfx = "_r" if i else ""
        pos = np.array(_tail((np.asarray(x, np.float64) for x in hist["ee_pos"][arm]), T))
        quat = np.array(_tail((np.asarray(x, np.float64) for x in hist["ee_quat"][arm]), T))
        rel = compute_relative_actions(pos, quat, T - 1)[:H]
        hand = np.stack(_tail((np.asarray(x, np.float64) for x in hist["hand"][arm]), T))[:H]
        cols["hand_qpos" + sfx] = hand.astype(np.float32)
        cols["ee_pos" + sfx] = rel[:, :3].astype(np.float32)
        cols["ee_quat" + sfx] = rel[:, 3:9].astype(np.float32)
        poses.append((pos[:H], quat[:H]))
    if rel_ee:
        (pos_l, quat_l), (pos_r, quat_r) = poses
        cols["rel_ee"] = relative_pose_9d(pos_l, quat_l, pos_r, quat_r).astype(np.float32)
    window = [{k: v[i].copy() for k, v in cols.items()} for i in range(H)]
    if hist.get("img") is not None:
        for step, img in zip(window, _tail((np.asarray(x) for x in hist["img"]), T)[:H]):
            step[WINDOW_IMAGES_KEY] = img
    return window
