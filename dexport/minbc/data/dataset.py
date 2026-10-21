"""Training dataset: zarr trajectories -> normalized observation windows and action chunks.

A sample is the window around one current frame f_c: ``obs_horizon`` observation frames ending
at f_c (``dp.obs_current_step``; otherwise the frames strictly before it) and the
``total_horizon`` strictly-future frames as the action chunk. Arm actions are the future
end-effector poses relative to f_c (9-D pos + rot6d per arm, arm blocks first), hand actions
the absolute 20-D targets per hand (commanded, or measured for ``data.hand_action_source=state``).
The EE observations are the observed frames' poses relative to f_c. Windows never span a
trajectory boundary or a Quest tracking jump (``data.jolt_split``); episode ends edge-pad.
"""

import numpy as np
import torch
import torch.utils.data

from dexport.common.pose import compute_relative_actions
from dexport.minbc.config import RAW_OBS_DIMS
from dexport.minbc.data import jolt_split
from dexport.minbc.data.zarr_io import ZarrFrameReader, count_steps, read_trajectory
from dexport.minbc.normalization import (
    REL_EE_STATS_KEYS,
    minmax_bounds,
    minmax_norm_data,
    normalize_obs,
    normalize_percentile,
    obs_clip_limit,
    percentile_bounds,
    set_identity_rot6d,
)

#: The arm-action and relative-EE bounds are fitted on every k-th window, about this many.
STATS_WINDOWS = 2000


def create_sample_indices(episode_ends, sequence_length, pad_before=0, pad_after=0):
    """``[buffer_start, buffer_end, sample_start, sample_end]`` for every window of every episode.

    A window may start ``pad_before`` frames before an episode and end ``pad_after`` frames past
    it; the missing rows are edge-padded by :func:`pad_sequence`.
    """
    indices = []
    for i in range(len(episode_ends)):
        start_idx = episode_ends[i - 1] if i > 0 else 0
        episode_length = episode_ends[i] - start_idx
        for idx in range(-pad_before, episode_length - sequence_length + pad_after + 1):
            buffer_start_idx = max(idx, 0) + start_idx
            buffer_end_idx = min(idx + sequence_length, episode_length) + start_idx
            start_offset = buffer_start_idx - (idx + start_idx)
            end_offset = (idx + sequence_length + start_idx) - buffer_end_idx
            indices.append(
                [buffer_start_idx, buffer_end_idx, start_offset, sequence_length - end_offset]
            )
    return np.array(indices)


def pad_sequence(data, sequence_length, sample_start_idx, sample_end_idx):
    """Place ``data`` at ``[sample_start_idx, sample_end_idx)`` of a window, repeating its edges."""
    if sample_start_idx == 0 and sample_end_idx >= sequence_length:
        return data
    out = np.zeros(shape=(sequence_length,) + data.shape[1:], dtype=data.dtype)
    if sample_start_idx > 0:
        out[:sample_start_idx] = data[0]
    if sample_end_idx < sequence_length:
        out[sample_end_idx:] = data[-1]
    out[sample_start_idx:sample_end_idx] = data
    return out


def sample_sequence(
    arrays, sequence_length, buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx
):
    """One window of every array of a dict: rows ``[buffer_start_idx, buffer_end_idx)``
    edge-padded by :func:`pad_sequence`."""
    return {
        key: pad_sequence(
            arr[buffer_start_idx:buffer_end_idx], sequence_length, sample_start_idx, sample_end_idx
        )
        for key, arr in arrays.items()
    }


class WindowDataset(torch.utils.data.Dataset):
    """Windows over a list of trajectories, for one split.

    Built without ``stats`` / ``obs_norm`` (the train split), it fits the normalization on its
    own trajectories: ``stats`` holds the action and relative-EE bounds, the merged
    ``stats["action"]`` = [arm blocks | hand blocks]; ``obs_norm`` the observation percentiles.
    The test split is built with the train split's. ``traj_sources`` tags each trajectory with
    the index of its '+'-joined ``train_data`` source (for the co-training sampler). A trajectory
    that cannot be read is skipped with a warning and listed in ``skipped``.
    """

    def __init__(
        self,
        config,
        traj_paths,
        split="train",
        traj_sources=None,
        stats=None,
        obs_norm=None,
        transform=None,
    ):
        data = config.data
        if (stats is None) != (obs_norm is None):
            raise ValueError("pass both stats and obs_norm (test split) or neither (train split)")
        if stats is None and split != "train":
            raise ValueError(f"the {split} split is normalized with the train split's stats")
        if data.hand_action_source not in ("command", "state"):
            raise ValueError(f"unknown data.hand_action_source {data.hand_action_source!r}")
        self.config = config
        self.split = split
        self.transform = transform
        self.data_key = tuple(data.data_key)
        self.bimanual = bool(data.bimanual)
        self.obs_horizon = config.dp.obs_horizon
        self.total_horizon = config.dp.total_horizon
        self.act_horizon = config.dp.act_horizon
        # Window rows: [obs frames | future frames]; base_idx is f_c, the frame both the arm
        # actions and the EE observations are expressed in.
        self.base_idx = self.obs_horizon - 1 if config.dp.obs_current_step else self.obs_horizon
        self.window_len = self.base_idx + 1 + self.total_horizon
        self.pad_before = self.base_idx
        self.pad_after = self.total_horizon - self.act_horizon
        self.stream_images = bool(data.im_stream) and "img" in self.data_key
        self._proprio = [k for k in self.data_key if k != "img"]
        self._arm_suffixes = ("", "_r") if self.bimanual else ("",)

        traj_paths = list(traj_paths)
        traj_sources = [0] * len(traj_paths) if traj_sources is None else list(traj_sources)
        if len(traj_sources) != len(traj_paths):
            raise ValueError("traj_sources must align 1:1 with traj_paths")
        self.num_sources = max(traj_sources) + 1 if traj_sources else 1

        self.skipped = []
        self.n_frames = self._count_frames(traj_paths)
        buffers, episode_ends, episode_sources = self._read_trajectories(traj_paths, traj_sources)
        self._normalize_observations(buffers, obs_norm)
        self.indices, self.source_ids = self._index_windows(episode_ends, episode_sources)
        if stats is None and len(self.indices) == 0:  # nothing to fit the action bounds on
            raise ValueError(
                f"no {split} windows: every episode is shorter than {self.min_episode_frames} "
                f"frames (after the tracking-jump splits)"
            )
        self.stats = self._normalize_actions(buffers, stats)
        self.buffers = buffers
        print(
            f"[dataset] {split}: {len(traj_paths) - len(self.skipped)} trajectories, "
            f"{len(buffers['action'])} frames, {len(self)} windows"
        )

    def _skip(self, traj, error):
        print(f"[dataset] WARNING: skipping unreadable trajectory {traj}: {error!r}")
        self.skipped.append(traj)

    def _count_frames(self, traj_paths):
        """Frames per trajectory; 0 for a trajectory without actions or one that cannot be read."""
        counts = []
        for traj in traj_paths:
            try:
                counts.append(count_steps(traj))
            except Exception as e:  # any unreadable trajectory is skipped
                self._skip(traj, e)
                counts.append(0)
        return counts

    def _read_trajectories(self, traj_paths, traj_sources):
        """Raw buffers of the kept frames of every readable trajectory, and the episode ends
        (trajectories split at tracking jumps) with their sources."""
        data = self.config.data
        n_rows = sum(self.n_frames)
        buffers = {k: np.zeros((n_rows, RAW_OBS_DIMS[k]), dtype=np.float32) for k in self._proprio}
        buffers["action"] = np.zeros((n_rows, data.hand_action_dim), dtype=np.float32)
        n_cam, height, width = len(data.im_key), data.im_height, data.im_width
        load_images = "img" in self.data_key and not self.stream_images
        if load_images:
            buffers["img"] = np.empty((n_rows, n_cam, 3, height, width), dtype=np.uint8)
        if self.stream_images:
            # Per buffer row: its frame index in, and the reader of, its trajectory.
            self._img_frame_idx = np.full(n_rows, -1, dtype=np.int64)
            self._img_store_idx = np.full(n_rows, -1, dtype=np.int64)
            self._img_stores = []
        self.jolt_stats = {"trajs_cut": 0, "cuts": 0, "runt_frames": 0}

        episode_ends, episode_sources, row = [], [], 0
        for traj, source, T in zip(traj_paths, traj_sources, self.n_frames):
            if T == 0:
                continue
            try:
                arrays = read_trajectory(traj, data, load_images=load_images)
                frame_shape = (n_cam, height, width, 3)
                if load_images and arrays["img"].shape[1:] != frame_shape:
                    raise ValueError(
                        f"images of shape {arrays['img'].shape[1:]}, not {frame_shape}"
                    )
                reader = ZarrFrameReader(traj) if self.stream_images else None
                if reader is not None:
                    reader.check_cameras(data.im_key, T, height, width)
            except Exception as e:  # any unreadable trajectory is skipped
                self._skip(traj, e)
                continue
            rows = slice(row, row + T)
            for k in self._proprio:
                buffers[k][rows] = arrays[k][:T]
            buffers["action"][rows] = self._hand_target(arrays)[:T]
            if load_images:
                buffers["img"][rows] = arrays["img"][:T].transpose(0, 1, 4, 2, 3)
            if reader is not None:
                self._img_frame_idx[rows] = np.arange(T)
                self._img_store_idx[rows] = len(self._img_stores)
                self._img_stores.append(reader)
            end = episode_ends[-1] if episode_ends else 0
            for length in self._segments(arrays, T):
                end += length
                episode_ends.append(end)
                episode_sources.append(source)
            row += T

        split = self.split
        if self.skipped:
            n = len(self.skipped)
            print(f"[dataset] {split}: skipped {n} of {len(traj_paths)} trajectories")
        if row == 0:
            raise RuntimeError(f"no {split} trajectory could be read from {len(traj_paths)} paths")
        if data.jolt_split:
            s = self.jolt_stats
            print(
                f"[jolt_split] {split}: {s['cuts']} cut(s) in {s['trajs_cut']} trajectories -> "
                f"{len(episode_ends)} episodes; {s['runt_frames']} frame(s) in segments shorter "
                f"than the {self.window_len}-frame window"
            )
        return {k: v[:row] for k, v in buffers.items()}, episode_ends, episode_sources

    def _hand_target(self, arrays):
        """Hand action rows: commanded targets, or the measured joints (state-as-action)."""
        if self.config.data.hand_action_source == "state":
            keys = ["hand_qpos"] + (["hand_qpos_r"] if self.bimanual else [])
            return np.concatenate([arrays[k] for k in keys], axis=1)
        return arrays["action"]

    def _segments(self, arrays, n):
        """Lengths of the episodes one ``n``-frame trajectory contributes (split at Quest
        tracking jumps)."""
        lengths = [n]
        data = self.config.data
        if data.jolt_split:
            streams = [arrays["ee_pos"][:n]]
            if self.bimanual:
                streams.append(arrays["ee_pos_r"][:n])
            cuts = jolt_split.find_cuts(
                streams,
                n,
                z_thresh=data.jolt_z_thresh,
                floor_m=data.jolt_floor_m,
                abs_gate_m=data.jolt_abs_gate_m,
            )
            if cuts:
                lengths, runt = jolt_split.split_lengths(cuts, n, self.window_len)
                self.jolt_stats["trajs_cut"] += 1
                self.jolt_stats["cuts"] += len(cuts)
                self.jolt_stats["runt_frames"] += runt
        return lengths

    def _normalize_observations(self, buffers, obs_norm):
        """Percentile-normalize the proprio buffers in place (fitting ``obs_norm`` when None).

        The absolute EE poses move to ``raw_ee`` unnormalized (samples use them relative to f_c);
        their percentiles are still stored in norm.pkl.
        """
        p = self.config.data.norm_percentile
        if obs_norm is None:
            obs_norm = {}
            for k in self._proprio:
                b = percentile_bounds(buffers[k], p)
                obs_norm[k] = {"lower": b["min"], "upper": b["max"]}
        self.raw_ee = {k: buffers.pop(k) for k in self._proprio if k in REL_EE_STATS_KEYS}
        self._clip = obs_clip_limit(p)
        for k in self._proprio:
            if k not in REL_EE_STATS_KEYS:
                bounds = obs_norm[k]
                buffers[k] = normalize_percentile(
                    buffers[k], bounds["lower"], bounds["upper"], self._clip
                )
        self.obs_norm = obs_norm

    def _index_windows(self, episode_ends, episode_sources):
        """The window index (see :func:`create_sample_indices`) and each window's source."""
        indices = create_sample_indices(
            episode_ends=episode_ends,
            sequence_length=self.window_len,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
        )
        if len(indices) == 0:
            return indices, np.zeros(0, dtype=np.int64)
        episode = np.searchsorted(np.asarray(episode_ends), indices[:, 0], side="right")
        return indices, np.asarray(episode_sources)[np.clip(episode, 0, len(episode_sources) - 1)]

    def _normalize_actions(self, buffers, stats):
        """Min/max-normalize the hand action buffer in place and return ``stats``.

        Without ``stats`` (the train split) they are fitted: the hand min/max of the raw buffer,
        and the arm-action and relative-EE percentiles of the windows' relative poses.
        """
        fit = stats is None
        if fit:
            stats = {"action": minmax_bounds(buffers["action"])}
        if self.config.data.hand_action_source == "state":
            self._assert_hand_target_moves(buffers["action"])
        hand_dim = buffers["action"].shape[1]
        buffers["action"] = minmax_norm_data(
            buffers["action"],
            stats["action"]["min"][-hand_dim:],
            stats["action"]["max"][-hand_dim:],
        )
        if not fit:
            return stats
        # Only the fitted (train) targets are clipped to the sampler's [-1, 1]: a dim that is
        # constant in the train split is divided by ~1e-8, so its test targets can be huge.
        np.clip(buffers["action"], -1.0, 1.0, out=buffers["action"])
        for suffix in self._arm_suffixes:
            arm, ee_pos, ee_quat = self._relative_pose_stats(
                self.raw_ee["ee_pos" + suffix], self.raw_ee["ee_quat" + suffix]
            )
            stats["arm_action" + suffix] = arm
            stats["ee_pos_rel" + suffix] = ee_pos
            stats["ee_quat_rel" + suffix] = ee_quat
        arms = [stats["arm_action" + suffix] for suffix in self._arm_suffixes]
        stats["action"] = {
            b: np.concatenate([arm[b] for arm in arms] + [stats["action"][b]])
            for b in ("min", "max")
        }
        return stats

    def _relative_pose_stats(self, raw_pos, raw_quat):
        """(arm_action, ee_pos_rel, ee_quat_rel) bounds from the relative poses of the windows."""
        p = self.config.data.norm_percentile
        future, observed = [], []
        poses = {"pos": raw_pos, "quat": raw_quat}
        for i in range(0, len(self.indices), max(1, len(self.indices) // STATS_WINDOWS)):
            w = sample_sequence(poses, self.window_len, *self.indices[i])
            rel = compute_relative_actions(w["pos"], w["quat"], self.base_idx)
            future.append(rel[self.base_idx + 1 :])
            observed.append(rel[: self.obs_horizon])
        future = np.concatenate(future, axis=0)
        observed = np.concatenate(observed, axis=0)
        arm = set_identity_rot6d(percentile_bounds(future.reshape(-1, 9), p), 3, 9)
        ee_pos = percentile_bounds(observed[:, :3], p)
        ee_quat = set_identity_rot6d(percentile_bounds(observed[:, 3:], p), 0, 6)
        return arm, ee_pos, ee_quat

    def _assert_hand_target_moves(self, hand):
        """A frozen measured channel as the action target would teach the hand to hold still."""
        a = np.asarray(hand, dtype=np.float64)
        dead = ~((np.nanmax(a, axis=0) - np.nanmin(a, axis=0)) > 1e-4)
        if dead.any():
            raise RuntimeError(
                f"data.hand_action_source='state': {int(dead.sum())}/{len(dead)} measured hand "
                f"dims never move across the {self.split} corpus (span <= 1e-4 rad), so the "
                f"policy would learn to hold them still. Find the trajectories with "
                f"scripts/check_frozen_channels.py and exclude them. Dead dims: "
                f"{np.flatnonzero(dead).tolist()}"
            )

    @property
    def min_episode_frames(self) -> int:
        """Length of the shortest episode that yields a window (edge padding included)."""
        return self.window_len - self.pad_before - self.pad_after

    def __len__(self):
        return len(self.indices)

    def _read_img_window(self, buffer_start, buffer_end, sample_start, sample_end):
        """(obs_horizon, n_cam, 3, H, W) uint8 obs frames read from the zarr store (im_stream)."""
        rows = np.arange(buffer_start, buffer_end, dtype=np.int64).reshape(-1, 1)
        rows = pad_sequence(rows, self.window_len, sample_start, sample_end)[: self.obs_horizon, 0]
        data = self.config.data
        out = np.empty(
            (self.obs_horizon, len(data.im_key), 3, data.im_height, data.im_width), dtype=np.uint8
        )
        for p, r in enumerate(rows):
            reader = self._img_stores[int(self._img_store_idx[r])]
            frame = int(self._img_frame_idx[r])
            for c, cam in enumerate(data.im_key):
                out[p, c] = reader.get(frame, cam).transpose(2, 0, 1)
        return out

    def _images(self, frames):
        """(To, n_cam, 3, H, W) frames -> float32 tensor through the transform (one call)."""
        x = torch.tensor(frames.astype(np.float32), dtype=torch.float32)
        shape = x.shape
        x = x.reshape(shape[0] * shape[1], *shape[2:])
        if self.transform is not None:
            x = self.transform(x)
        return x.reshape(shape[:3] + (x.shape[-2], x.shape[-1]))

    def __getitem__(self, idx):
        window = self.indices[idx]
        win = sample_sequence(self.buffers, self.window_len, *window)
        ee = sample_sequence(self.raw_ee, self.window_len, *window)
        b, To = self.base_idx, self.obs_horizon
        arm_blocks, ee_obs = [], {}
        for suffix in self._arm_suffixes:
            rel = compute_relative_actions(ee["ee_pos" + suffix], ee["ee_quat" + suffix], b)
            arm = self.stats["arm_action" + suffix]
            arm_blocks.append(
                np.clip(minmax_norm_data(rel[b + 1 :], arm["min"], arm["max"]), -1, 1).astype(
                    np.float32
                )
            )
            for key, part in (("ee_pos", rel[:To, :3]), ("ee_quat", rel[:To, 3:])):
                ee_obs[key + suffix] = normalize_obs(
                    key + suffix, part, self.stats, self.obs_norm, self._clip
                ).astype(np.float32)

        # Keys in data_key order, then action; streamed images are read last.
        sample = {}
        for k in self.data_key:
            if k == "img":
                if not self.stream_images:
                    sample["img"] = self._images(win["img"][:To])
            elif k in ee_obs:
                sample[k] = torch.tensor(ee_obs[k], dtype=torch.float32)
            else:
                sample[k] = torch.tensor(win[k][:To], dtype=torch.float32)
        hand = win["action"][b + 1 :]
        action = np.concatenate(arm_blocks + [hand], axis=-1).astype(np.float32)
        sample["action"] = torch.tensor(action, dtype=torch.float32)
        if self.stream_images:
            sample["img"] = self._images(self._read_img_window(*window))
        return sample
