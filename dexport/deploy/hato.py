"""HATO action aggregation (Lin et al., arXiv:2404.16823): overlapping chunks, averaged per tick.

Every finished chunk is tagged with the control tick ``p`` of the observation it was computed from;
row ``i`` is the command for tick ``p + i``. Rows are filed into per-tick buckets and rows whose
tick has already passed are dropped (the latency compensation). The command for a tick is the
average of every chunk that predicted it; a tick nobody predicted is held: the arms get no new
waypoint and the hands are re-sent their last targets.

The policy's arm action is relative to its own chunk's base pose, so chunks are averaged after
conversion to absolute flange poses: positions and finger targets arithmetically, quaternions after
aligning each to the hemisphere of the oldest vote (q and -q are the same rotation) and normalizing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np

from dexport.common.pose import mean_quat


@dataclass
class ChunkTargets:
    """One chunk as absolute targets: ``arm_pos`` (rows, 3), ``arm_quat`` (rows, 4) xyzw and
    ``hand`` (rows, 20) per arm, plus the policy's own rows (``policy_actions``, logged only).
    Row ``i`` is the command for control step ``launch_step + i``."""

    launch_step: int
    arm_pos: dict
    arm_quat: dict
    hand: dict
    policy_actions: np.ndarray


@dataclass
class StepCommand:
    """The averaged command of one control step. ``policy_action`` is the newest vote's row of
    the policy's output (its arm block is relative to that chunk's own base, so policy rows are
    never averaged)."""

    arm_pos: dict
    arm_quat: dict
    hand: dict
    policy_action: np.ndarray
    n_votes: int
    vote_launch_steps: list = field(default_factory=list)


class _Vote(NamedTuple):
    """One chunk's row for one control step: the per-arm targets and the policy's own row."""

    launch_step: int
    arm_pos: dict
    arm_quat: dict
    hand: dict
    policy_action: np.ndarray


class ChunkAggregator:
    """Per-tick buckets of chunk rows, keyed by absolute control step."""

    def __init__(self):
        #: {step: [_Vote, ...]} in filing order
        self._buckets: dict[int, list[_Vote]] = {}

    def add(self, targets: ChunkTargets, now_step: int) -> int:
        """File a chunk's rows by absolute step, dropping rows before ``now_step``. Returns the
        number of rows filed."""
        p = int(targets.launch_step)
        filed = 0
        for i in range(targets.policy_actions.shape[0]):
            step = p + i
            if step < now_step:
                continue
            self._buckets.setdefault(step, []).append(
                _Vote(
                    launch_step=p,
                    arm_pos={a: targets.arm_pos[a][i] for a in targets.arm_pos},
                    arm_quat={a: targets.arm_quat[a][i] for a in targets.arm_quat},
                    hand={a: targets.hand[a][i] for a in targets.hand},
                    policy_action=targets.policy_actions[i],
                )
            )
            filed += 1
        self._drop_before(now_step)
        return filed

    def _drop_before(self, step: int) -> None:
        for s in [s for s in self._buckets if s < step]:
            del self._buckets[s]

    def take(self, step: int) -> StepCommand | None:
        """The averaged command for ``step`` (its bucket is consumed), or None if no chunk
        predicted it."""
        self._drop_before(step)
        votes = self._buckets.pop(step, None)
        if not votes:
            return None
        arms = list(votes[0].arm_pos)
        if len(votes) == 1:
            (v,) = votes
            pos = {a: np.asarray(v.arm_pos[a], dtype=np.float64) for a in arms}
            quat = {a: np.asarray(v.arm_quat[a], dtype=np.float64) for a in arms}
            hand = {a: np.asarray(v.hand[a], dtype=np.float64) for a in arms}
        else:
            pos = {
                a: np.mean(np.asarray([v.arm_pos[a] for v in votes], np.float64), axis=0)
                for a in arms
            }
            quat = {a: mean_quat([v.arm_quat[a] for v in votes]) for a in arms}
            hand = {
                a: np.mean(np.asarray([v.hand[a] for v in votes], np.float64), axis=0) for a in arms
            }
        return StepCommand(
            arm_pos=pos,
            arm_quat=quat,
            hand=hand,
            policy_action=np.asarray(votes[-1].policy_action, dtype=np.float64),
            n_votes=len(votes),
            vote_launch_steps=[int(v.launch_step) for v in votes],
        )
