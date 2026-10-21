"""Zarr recorder of collected episodes: one :class:`EpisodeRecorder` and one store per session.

Layout: ``<save_dir>/<MMDD_HHMM>[_<tag>]/trajectories.zarr/traj_<i>/<channel>``, ``i`` counting
saved episodes only. Channels, one row per recorded tick: ``timestamp`` (T,); ``action`` /
``action_r`` (commanded) and ``hand_joints`` / ``hand_joints_r`` (measured) (T, 5, 4) float64
joints of the left / right hand, in its hardware convention; ``pose_l`` / ``pose_r`` (T, 3) and
``quat_l`` / ``quat_r`` (T, 4, xyzw) Quest controller poses, each when enabled; one (T, H, W, 3)
uint8 RGB channel per camera key; ``hand_joints_age`` / ``hand_joints_r_age`` (T,) readback ages
(s), when enabled. The names are :mod:`dexport.common.channels`. Arrays use the zarr v2 defaults
(Blosc lz4).
"""

from __future__ import annotations

import glob
import os
import time

import numpy as np
import zarr
from scipy.spatial.transform import Rotation

from dexport.collect.channel_health import diagnose_hand_channels
from dexport.common.channels import CHANNELS, TIMESTAMP

# A readback at least this old (s) is stale: the driver returned a held pose. A healthy Ethernet
# readback is a few ms old, so 0.1 s is a stall, not jitter.
HAND_STALE_AGE_S = 0.1

_LABEL = {"left": "Left", "right": "Right"}
#: Key of each side's controller in ``VRController.poses``.
_QUEST_KEY = {"left": "l", "right": "r"}


class EpisodeRecorder:
    """Buffers one episode at a time and appends it to the session's store on :meth:`stop`.

    The policy loader indexes every channel by step, so every channel gets exactly one row per
    tick: a momentarily missing controller pose, hand value or camera frame repeats the last one,
    and a value never seen in the episode is NaN. An episode whose channels still differ in length
    (a camera that missed its first frames) is not saved; neither it nor a discarded episode uses
    up an index. A session directory whose store already holds episodes (a second session started
    in the same minute with the same tag) is refused.
    """

    def __init__(
        self,
        save_dir: str = "data",
        *,
        tag: str = "",
        image_keys=(),
        save_pose_l: bool = True,
        save_quat_l: bool = True,
        save_pose_r: bool = False,
        save_quat_r: bool = False,
        save_readback_age: bool = False,
    ):
        self._image_keys = list(image_keys)
        session = time.strftime("%m%d_%H%M")
        if tag:
            session = f"{session}_{tag}"
        self._save_dir = os.path.join(save_dir, session)
        self._save_pose = {"l": save_pose_l, "r": save_pose_r}
        self._save_quat = {"l": save_quat_l, "r": save_quat_r}
        self._save_readback_age = save_readback_age

        self._zarr_path = os.path.join(self._save_dir, "trajectories.zarr")
        if glob.glob(os.path.join(glob.escape(self._zarr_path), "traj_*")):
            raise FileExistsError(
                f"{self._zarr_path} already holds episodes (a session with the same tag started "
                "this minute); wait a minute or set another trajectory.tag"
            )
        self._store: zarr.Group | None = None  # opened on the first save
        print(f"[recorder] Saving to {self._zarr_path}")

        self._traj_count = 0
        self._buffer: dict[str, list] = {}
        self._recording = False
        self._reset_episode_state()

    @property
    def recording(self) -> bool:
        return self._recording

    @property
    def next_index(self) -> int:
        """Index the next saved episode will get (``traj_<next_index>``)."""
        return self._traj_count

    def start(self) -> None:
        self._buffer = self._make_empty_buffer()
        self._reset_episode_state()
        self._recording = True
        print("[recorder] ▶  Recording started.")

    def discard(self) -> None:
        """Drop the current episode unsaved (the operator marked it failed)."""
        if not self._recording:
            return
        self._recording = False
        self._buffer = self._make_empty_buffer()
        print("[recorder] ✗  Trajectory discarded (marked failed) — not saved.")

    def step(
        self,
        poses: dict,
        *,
        measured: dict,
        commands: dict,
        ages: dict | None = None,
        images: dict | None = None,
    ) -> None:
        """Append one tick (no-op unless recording).

        ``poses`` is ``VRController.poses`` (``{"l"|"r": 4x4}``). ``measured`` / ``commands`` map
        ``"left"`` / ``"right"`` to that hand's (5, 4) measured / commanded joints (a missing side
        or None repeats the episode's last value); ``ages`` maps a side to its readback age in s
        (missing or None: stored as NaN); ``images`` maps camera keys to frames (None: no new one).
        """
        if not self._recording:
            return
        buf = self._buffer
        buf[TIMESTAMP].append(time.time())
        for side, ch in CHANNELS.items():
            buf[ch.measured].append(self._hand_value(ch.measured, measured.get(side)))
            buf[ch.command].append(self._hand_value(ch.command, commands.get(side)))
        if self._save_readback_age:
            for side, ch in CHANNELS.items():
                age = (ages or {}).get(side)
                buf[ch.age].append(np.float64(age if age is not None else np.nan))

        for side, ch in CHANNELS.items():
            q = _QUEST_KEY[side]
            if not (self._save_pose[q] or self._save_quat[q]):
                continue
            pos, quat, status = self._pose_quat_for_side(poses, q)
            self._track_counts[q][status] += 1
            if self._save_pose[q]:
                buf[ch.pos].append(pos)
            if self._save_quat[q]:
                buf[ch.quat].append(quat)

        frames = images or {}
        for key in self._image_keys:
            frame = frames.get(key)
            if frame is not None:
                self._last_image[key] = frame
            # Until a camera's first frame arrives its channel stays short (caught in stop()).
            if self._last_image[key] is not None:
                buf[key].append(self._last_image[key].copy())

    def stop(self) -> None:
        """Save the current episode and print its health report."""
        if not self._recording:
            return
        self._recording = False

        n_steps = next((len(v) for v in self._buffer.values() if v), 0)
        if n_steps == 0:
            print("[recorder] ⏹  Empty trajectory discarded.")
            return
        lengths = {k: len(v) for k, v in self._buffer.items() if v}
        if len(set(lengths.values())) > 1:
            print(
                f"[recorder] ⚠️  Channel length mismatch {lengths} — "
                "trajectory NOT saved (it could not be loaded)."
            )
            return

        traj_name = f"traj_{self._traj_count}"
        self._save_to_zarr(traj_name)
        print(f"[recorder] ⏹  Saved {traj_name}  ({n_steps} steps).")
        self._report()
        self._traj_count += 1

    def _reset_episode_state(self) -> None:
        self._last_pose: dict[str, np.ndarray | None] = {"l": None, "r": None}
        self._last_quat: dict[str, np.ndarray | None] = {"l": None, "r": None}
        self._track_counts = {s: {"live": 0, "held": 0, "nan": 0} for s in ("l", "r")}
        self._last_image: dict[str, np.ndarray | None] = {k: None for k in self._image_keys}
        hand_keys = [k for ch in CHANNELS.values() for k in (ch.measured, ch.command)]
        self._last_hand: dict[str, np.ndarray | None] = dict.fromkeys(hand_keys)
        self._hand_counts = {k: {"live": 0, "held": 0, "nan": 0} for k in hand_keys}

    def _pose_quat_for_side(self, poses: dict, side: str):
        """``(position, xyzw quaternion, status)`` with status ``live`` (tracked this tick),
        ``held`` (repeats the last pose of this episode) or ``nan`` (not seen yet)."""
        if side in poses:
            mat = np.asarray(poses[side])
            pos = mat[:3, 3].copy()
            quat = Rotation.from_matrix(mat[:3, :3]).as_quat()
            self._last_pose[side] = pos
            self._last_quat[side] = quat
            return pos, quat, "live"
        if self._last_pose[side] is not None:
            return self._last_pose[side].copy(), self._last_quat[side].copy(), "held"
        return np.full(3, np.nan), np.full(4, np.nan), "nan"

    def _hand_value(self, key: str, value: np.ndarray | None) -> np.ndarray:
        if value is not None:
            v = np.asarray(value, dtype=np.float64).copy()
            self._last_hand[key] = v
            self._hand_counts[key]["live"] += 1
            return v
        if self._last_hand[key] is not None:
            self._hand_counts[key]["held"] += 1
            return self._last_hand[key].copy()
        self._hand_counts[key]["nan"] += 1
        return np.full((5, 4), np.nan, dtype=np.float64)

    def _report(self) -> None:
        """Collection-time warnings for the episode just saved."""
        self._report_undriven_hands()
        self._report_channel_health()
        self._report_stale_readback()
        self._report_carried_forward()
        self._report_tracking()
        self._report_distance()

    def _report_undriven_hands(self) -> None:
        for side, ch in CHANNELS.items():
            cmd = self._buffer.get(ch.command)
            if cmd and np.all(np.isnan(np.stack(cmd))):
                print(
                    f"[recorder] ℹ️  {_LABEL[side]} hand recorded but never driven "
                    f"(all {len(cmd)} steps NaN) — no {side}-hand action supplied."
                )

    def _report_channel_health(self) -> None:
        hand_keys = {k for ch in CHANNELS.values() for k in (ch.measured, ch.command)}
        chans = {k: np.stack(v) for k, v in self._buffer.items() if v and k in hand_keys}
        for f in diagnose_hand_channels(chans):
            mark = "⚠️" if f["level"] == "warn" else "ℹ️"
            print(f"[recorder] {mark}  {f['message']}")

    def _report_stale_readback(self) -> None:
        if not self._save_readback_age:
            return
        for side, ch in CHANNELS.items():
            key = ch.age
            a = np.asarray(self._buffer.get(key) or [], dtype=np.float64)
            finite = a[np.isfinite(a)]
            n_stale = int(np.sum(finite >= HAND_STALE_AGE_S)) if finite.size else 0
            if n_stale:
                print(
                    f"[recorder] ⚠️  {_LABEL[side]} readback STALE on {n_stale}/{len(a)} ticks "
                    f"(max age {finite.max():.2f}s ≥ {HAND_STALE_AGE_S:.2f}s) — held/zero "
                    f"proprio there (the per-tick age is recorded in {key})."
                )

    def _report_carried_forward(self) -> None:
        for key, counts in self._hand_counts.items():
            total = sum(counts.values())
            if not total or not (counts["held"] or counts["nan"]) or counts["nan"] == total:
                continue  # all live, or never driven (reported above)
            print(
                f"[recorder] {key}: {counts['live']} live, "
                f"{counts['held']} held (carried forward), {counts['nan']} nan "
                f"({100.0 * counts['live'] / total:.0f}% live)."
            )

    def _report_tracking(self) -> None:
        for side, q in _QUEST_KEY.items():
            if not (self._save_pose[q] or self._save_quat[q]):
                continue
            label = _LABEL[side]
            c = self._track_counts[q]
            total = c["live"] + c["held"] + c["nan"] or 1
            if c["nan"] == total:
                print(
                    f"[recorder] ⚠️  {label} controller NEVER tracked "
                    f"(all {total} steps NaN) — check the controller / pose key."
                )
            elif c["held"] or c["nan"]:
                print(
                    f"[recorder] {label} tracking: {c['live']} live, "
                    f"{c['held']} held (dropout), {c['nan']} nan "
                    f"({100.0 * c['live'] / total:.0f}% live)."
                )

    def _report_distance(self) -> None:
        for side, ch in CHANNELS.items():
            values = self._buffer.get(ch.pos)
            if values and len(values) >= 2:
                dist_cm = np.linalg.norm(values[-1] - values[0]) * 100.0  # NaN if never tracked
                print(f"[recorder] {_LABEL[side]} start-to-end distance: {dist_cm:.2f} cm")

    def _make_empty_buffer(self) -> dict[str, list]:
        keys = [TIMESTAMP]
        keys += [k for ch in CHANNELS.values() for k in (ch.measured, ch.command)]
        for side, ch in CHANNELS.items():
            keys += [ch.pos] if self._save_pose[_QUEST_KEY[side]] else []
            keys += [ch.quat] if self._save_quat[_QUEST_KEY[side]] else []
        if self._save_readback_age:
            keys += [ch.age for ch in CHANNELS.values()]
        return {k: [] for k in keys + self._image_keys}

    def _save_to_zarr(self, traj_name: str) -> None:
        os.makedirs(self._save_dir, exist_ok=True)
        if self._store is None:
            self._store = zarr.open_group(self._zarr_path, mode="a")
        grp = self._store.require_group(traj_name)
        for key, values in self._buffer.items():
            if not values:
                continue
            if key in grp:
                del grp[key]
            grp.create_dataset(key, data=np.stack(values))
