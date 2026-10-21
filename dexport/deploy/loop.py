"""The asynchronous HATO control loop: 30 Hz control, inference on a worker thread.

Each tick: read the observation; file any finished chunk's first ``chunk_rows`` rows by absolute
tick (rows already in the past are dropped); launch the next inference as soon as the worker is
free, with the flange pose commanded at the previous tick as the new chunk's base pose; take the
tick's averaged command (:mod:`dexport.deploy.hato`), solve IK and schedule one joint waypoint per
arm for the next tick; send the hand targets. A tick no chunk covers is held. Arm waypoints are
stamped on a fixed step grid (:class:`StepClock`) that is re-anchored only when the loop overruns.
Warm-up inferences and the latency check run before the first tick (``_Episode.prime``).

Space pauses (the arm is held, the hands are re-sent their targets, the observation history keeps
filling) and resumes; Esc ends the episode, and so does any key that sends an escape sequence,
such as an arrow key.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from dexport.common.terminal import ESC, KeyPoller
from dexport.common.timing import precise_wait
from dexport.deploy.errors import DeployError
from dexport.deploy.hato import ChunkAggregator, ChunkTargets, StepCommand
from dexport.deploy.obs import new_history
from dexport.deploy.worker import AsyncChunkWorker, InferRequest, InferResult

#: A worker busy for longer than max(_STALL_FLOOR_TICKS, _STALL_FACTOR x the longest recent chunk
#: delay) is stalled (e.g. hung in a CUDA call): end the episode instead of holding forever.
_STALL_FACTOR = 6
_STALL_FLOOR_TICKS = 40
_PRIMING_TIMEOUT_S = 120.0
#: A waypoint is stamped at least this far ahead, so it is never already due when it is queued.
_ARRIVAL_MARGIN_S = 2e-3
#: Step-grid overruns reported per episode (each re-anchors the grid; later ones only count).
_MAX_OVERRUN_PRINTS = 5
#: Held-tick counts at which the starvation warning is printed (and then every 200th).
_HELD_WARNINGS = (1, 10, 100)


class LatencyError(DeployError):
    """Inference is too slow for the chunk length: HATO needs 2 x latency <= chunk rows."""


def shielded(what: str, fn) -> None:
    """Run ``fn()`` during teardown: an exception is printed, not raised, so it cannot mask the
    error that ended the episode."""
    try:
        fn()
    except BaseException as e:  # teardown must not mask the original error
        print(f"[deploy] {what} failed: {e!r}", flush=True)


class StepClock:
    """Control step -> monotonic time, ``t0 + step * dt``. ``check(step)`` re-anchors the grid when
    the step is already overdue by more than ``tolerance_ticks`` and returns the overrun (s)."""

    def __init__(self, dt: float, tolerance_ticks: float = 1.0):
        self.dt = float(dt)
        self.tolerance = float(tolerance_ticks) * self.dt
        self.t0 = time.monotonic()
        self.n_reanchors = 0

    def time_of(self, step: int) -> float:
        return self.t0 + step * self.dt

    def check(self, step: int) -> float:
        overrun = time.monotonic() - self.time_of(step)
        if overrun > self.tolerance:
            self.t0 = time.monotonic() - step * self.dt
            self.n_reanchors += 1
            return overrun
        return 0.0


@dataclass
class ArmIO:
    """One arm's bindings: ``get_ee_pose() -> (pos, quat_xyzw)`` (measured flange pose),
    ``solve_ik(pos, quat) -> joints``, ``schedule_joints(joints, t_arrive_monotonic)``, the arm's
    and hand's columns of the policy action (``slice``), its mount (4x4) and its hand
    (``get_qpos_20`` / ``send_qpos_20``)."""

    get_ee_pose: Callable
    solve_ik: Callable
    schedule_joints: Callable
    arm_slice: slice
    hand_slice: slice
    mount: np.ndarray
    hand: object


@dataclass
class LoopIO:
    """The embodiment: ``arms`` (ordered ``{arm: ArmIO}``), ``build_obs_window(hist)`` (the
    policy-frame observation window from the loop's history, :func:`dexport.deploy.obs.
    new_history`), ``stop_motion()`` (drop queued arm motion and hold) and ``get_image_stack()``
    ((n_cam, H, W, 3) uint8, or None without images)."""

    arms: dict
    build_obs_window: Callable
    stop_motion: Callable
    get_image_stack: Callable | None = None


@dataclass
class LoopConfig:
    frequency: float  # Hz
    obs_horizon: int
    n_steps: int
    chunk_rows: int  # rows of each chunk that vote
    warmup_inferences: int
    fail_on_hold: bool = False  # a held tick is an error (replay: it would offset the rest)


@dataclass
class EpisodeStats:
    n_steps: int = 0
    n_launches: int = 0
    n_chunks: int = 0  # chunks that filed at least one row
    n_stale_chunks: int = 0  # chunks that arrived entirely in the past
    n_held: int = 0  # ticks without a command
    n_reanchors: int = 0
    latency_ticks: list = field(default_factory=list)
    votes_per_tick: list = field(default_factory=list)  # of the ticks with a command

    def summary(self) -> str:
        lat = np.asarray(self.latency_ticks or [0])
        votes = np.asarray(self.votes_per_tick or [0])
        return (
            f"{self.n_steps} steps, {self.n_launches} inferences, {self.n_chunks} chunks used, "
            f"{self.n_stale_chunks} stale, {self.n_held} held ticks; latency median "
            f"{np.median(lat):.0f} / max {lat.max()} ticks; {votes.mean():.1f} votes per tick; "
            f"{self.n_reanchors} step-grid re-anchors"
        )


@dataclass
class _Command:
    """The command in force, per arm: flange position, xyzw quaternion and hand target."""

    pos: dict
    quat: dict
    hand: dict

    @classmethod
    def measured(cls, io: LoopIO) -> "_Command":
        """The measured start pose and hand state."""
        pos, quat, hand = {}, {}, {}
        for a, arm in io.arms.items():
            p, q = arm.get_ee_pose()
            pos[a] = np.asarray(p, dtype=np.float64).copy()
            quat[a] = np.asarray(q, dtype=np.float64).copy()
            hand[a] = np.asarray(arm.hand.get_qpos_20(), dtype=np.float64).copy()
        return cls(pos, quat, hand)

    def update(self, cmd: StepCommand) -> None:
        for a in self.pos:
            self.pos[a] = np.asarray(cmd.arm_pos[a], dtype=np.float64)
            self.quat[a] = np.asarray(cmd.arm_quat[a], dtype=np.float64)
            self.hand[a] = np.asarray(cmd.hand[a], dtype=np.float64)


class _Episode:
    """One episode's state and steps; :func:`run_episode` drives it."""

    def __init__(self, predict, io: LoopIO, cfg: LoopConfig, records: list, stats: EpisodeStats):
        self.io, self.cfg, self.records, self.stats = io, cfg, records, stats
        self.arms = list(io.arms)
        self.dt = 1.0 / float(cfg.frequency)
        self.rows = int(cfg.chunk_rows)
        n_hist = int(cfg.obs_horizon) + 1
        self.hist = new_history(self.arms, n_hist, io.get_image_stack is not None)
        self.command = _Command.measured(io)  # in force until the first tick overwrites it
        self.worker = AsyncChunkWorker(
            predict,
            arms=self.arms,
            arm_slices={a: io.arms[a].arm_slice for a in self.arms},
            hand_slices={a: io.arms[a].hand_slice for a in self.arms},
            mounts={a: io.arms[a].mount for a in self.arms},
            control_dt=self.dt,
        )
        self.agg = ChunkAggregator()
        self.keys = KeyPoller()
        self.clock: StepClock | None = None  # started after priming
        self.delays: deque = deque(maxlen=10)  # ticks from launch to filing, recent chunks
        self.last_launch = 0
        self.last_policy_action = None
        self.t_status = -1.0

    # observations, requests, hands
    def read_obs(self) -> None:
        h = self.hist
        for a in self.arms:
            p, q = self.io.arms[a].get_ee_pose()
            h["ee_pos"][a].append(np.asarray(p, dtype=np.float64).copy())
            h["ee_quat"][a].append(np.asarray(q, dtype=np.float64).copy())
            h["hand"][a].append(np.asarray(self.io.arms[a].hand.get_qpos_20(), np.float32).copy())
        if h["img"] is not None:
            h["img"].append(np.asarray(self.io.get_image_stack()).copy())

    def request(self, step: int) -> InferRequest:
        """The inference job for ``step``: the observation window (built and copied here, so the
        worker never iterates the live history) and the command in force as the base pose."""
        window = [
            {k: (np.asarray(v).copy() if isinstance(v, np.ndarray) else v) for k, v in s.items()}
            for s in self.io.build_obs_window(self.hist)
        ]
        c = self.command
        return InferRequest(
            launch_step=step,
            obs_window=window,
            base_pos={a: c.pos[a].copy() for a in self.arms},
            base_quat={a: c.quat[a].copy() for a in self.arms},
        )

    def targets(self, res: InferResult) -> ChunkTargets:
        r = self.rows
        return ChunkTargets(
            launch_step=int(res.launch_step),
            arm_pos={a: res.arm_pos[a][:r] for a in self.arms},
            arm_quat={a: res.arm_quat[a][:r] for a in self.arms},
            hand={a: res.hand[a][:r] for a in self.arms},
            policy_actions=res.chunk[:r],
        )

    def send_hands(self) -> None:
        for a in self.arms:
            self.io.arms[a].hand.send_qpos_20(self.command.hand[a])

    # priming
    def infer_now(self, n: int) -> InferResult:
        """Warm-up inference ``n``: a blocking inference for tick 0 on the still robot."""
        self.read_obs()
        self.worker.submit(self.request(0))
        t_wait = time.monotonic()
        while (res := self.worker.poll()) is None:
            self.worker.raise_if_failed()
            if time.monotonic() - t_wait > _PRIMING_TIMEOUT_S:
                raise RuntimeError(f"inference did not return within {_PRIMING_TIMEOUT_S} s")
            time.sleep(0.002)
        print(
            f"[deploy] warm-up inference {n}: {res.latency_s * 1000:.0f} ms "
            f"= {res.latency_ticks} ticks at {self.cfg.frequency:g} Hz",
            flush=True,
        )
        return res

    def prime(self) -> InferResult:
        """Fill the history and run ``warmup_inferences`` synchronous inferences on the still
        robot (the first CUDA call is a cold start); returns the last one's result, which fills
        the first ticks. Gapless coverage needs a latency of ``L`` ticks with
        ``2L <= chunk_rows``, so a slower one raises :class:`LatencyError` (a too slow last
        warm-up is measured once more first)."""
        for _ in range(int(self.cfg.obs_horizon) + 1):
            self.read_obs()
        n = max(1, int(self.cfg.warmup_inferences))
        for w in range(n):
            res = self.infer_now(w + 1)
        if 2 * res.latency_ticks > self.rows:  # one slow warm-up may be a spike: measure again
            res = self.infer_now(n + 1)
            if 2 * res.latency_ticks > self.rows:
                raise LatencyError(
                    f"inference takes {res.latency_ticks} ticks at {self.cfg.frequency:g} Hz "
                    f"(twice in a row), but the {self.rows}-row chunks need 2 x latency <= "
                    f"{self.rows}: use fewer diffusion steps or a faster GPU"
                )
        return res

    def start(self) -> None:
        """Prime, file the last warm-up's chunk at tick 0 and start the step grid."""
        first = self.prime()
        self.stats.latency_ticks.append(int(first.latency_ticks))
        self.clock = StepClock(self.dt)
        self.stats.n_launches += 1
        self.agg.add(self.targets(first), now_step=0)
        self.stats.n_chunks += 1
        self.last_policy_action = np.asarray(first.chunk[0], dtype=np.float64)
        print(
            f"[deploy] running {self.cfg.n_steps} steps at {self.cfg.frequency:g} Hz: Space "
            "pauses, Esc (or an arrow key) ends the episode.",
            flush=True,
        )

    # one tick
    def file_finished(self, k: int) -> bool:
        """File a finished chunk, if any; True if it filed rows (a new chunk joins the average)."""
        res = self.worker.poll()
        if res is None:
            return False
        self.stats.latency_ticks.append(int(res.latency_ticks))
        self.delays.append(k - int(res.launch_step))
        if self.agg.add(self.targets(res), now_step=k):
            self.stats.n_chunks += 1
            return True
        self.stats.n_stale_chunks += 1
        return False

    def maybe_launch(self, k: int) -> None:
        """Launch the next inference into an idle worker, at most one per tick; raise if the
        worker has been busy for far longer than recent chunks took."""
        if self.worker.idle_and_drained():
            if k - self.last_launch >= 1:
                self.worker.submit(self.request(k))
                self.last_launch = k
                self.stats.n_launches += 1
        elif self.worker.busy:
            budget = max(_STALL_FLOOR_TICKS, _STALL_FACTOR * max(self.delays, default=1))
            if k - self.last_launch > budget:
                raise RuntimeError(
                    f"the inference worker has been busy for {k - self.last_launch} ticks "
                    f"(> {budget}) without returning: it is stalled"
                )

    def hold(self, k: int) -> None:
        self.stats.n_held += 1
        if self.cfg.fail_on_hold:
            raise RuntimeError(f"no command for step {k}: the inference fell behind")
        n = self.stats.n_held
        if n in _HELD_WARNINGS or n % 200 == 0:
            print(
                f"[deploy] WARNING: no chunk covers step {k} ({n} held ticks): inference is too "
                "slow for the chunk length.",
                flush=True,
            )

    def command_arms(self, k: int) -> None:
        """IK for every arm first, so all of the tick's waypoints share one arrival stamp."""
        c = self.command
        joints = {a: self.io.arms[a].solve_ik(c.pos[a], c.quat[a]) for a in self.arms}
        t_arrive = max(self.clock.time_of(k + 1), time.monotonic() + _ARRIVAL_MARGIN_S)
        for a in self.arms:
            self.io.arms[a].schedule_joints(joints[a], t_arrive)

    def print_status(self, k: int, cmd: StepCommand | None) -> None:
        """About once a second: the step, its votes and the first arm's commanded position."""
        if time.monotonic() - self.t_status < 1.0:
            return
        self.t_status = time.monotonic()
        pos = ", ".join(f"{v:+.3f}" for v in self.command.pos[self.arms[0]])
        print(
            f"[deploy] step {k:4d}  votes {0 if cmd is None else cmd.n_votes}  "
            f"flange [{pos}]" + ("  HELD" if cmd is None else ""),
            flush=True,
        )

    def record(self, k: int, cmd: StepCommand | None, new_chunk: bool) -> dict:
        """The tick's log entry: what was commanded and what was measured."""
        c, h = self.command, self.hist
        rec = {
            "step": k,
            "t": time.time(),
            "held": cmd is None,
            "n_votes": 0 if cmd is None else int(cmd.n_votes),
            "new_chunk": new_chunk,
            "vote_launch_steps": [] if cmd is None else list(cmd.vote_launch_steps),
            "policy_action": np.asarray(self.last_policy_action, dtype=np.float64).copy(),
        }
        for a in self.arms:
            rec[f"cmd_pos_{a}"] = c.pos[a].copy()
            rec[f"cmd_quat_{a}"] = c.quat[a].copy()
            rec[f"cmd_hand_{a}"] = c.hand[a].copy()
            rec[f"obs_pos_{a}"] = np.asarray(h["ee_pos"][a][-1], np.float64).copy()
            rec[f"obs_quat_{a}"] = np.asarray(h["ee_quat"][a][-1], np.float64).copy()
            rec[f"obs_hand_{a}"] = np.asarray(h["hand"][a][-1], np.float64).copy()
        return rec

    def pause(self, k: int) -> bool:
        """Hold the arms until Space (False: resume) or Esc (True: end the episode), re-sending the
        hands their targets and filling the observation history every tick."""
        print(f"[deploy] paused at step {k}: Space resumes, Esc ends the episode.", flush=True)
        self.io.stop_motion()
        while True:
            time.sleep(self.dt)
            self.worker.raise_if_failed()
            self.read_obs()
            self.send_hands()
            key = self.keys.read()
            if ESC in key:
                return True
            if " " in key:
                break
        print(f"[deploy] resumed at step {k}.", flush=True)
        self.clock.check(k)
        return False

    def tick(self, k: int) -> bool:
        """Control step ``k``; False when Esc ends the episode."""
        self.worker.raise_if_failed()
        key = self.keys.read()
        if ESC in key or (" " in key and self.pause(k)):
            print(f"\n[deploy] Esc: episode ended at step {k}.", flush=True)
            return False
        overrun = self.clock.check(k)
        if overrun and self.clock.n_reanchors <= _MAX_OVERRUN_PRINTS:
            print(f"[deploy] step {k} overran by {overrun * 1000:.0f} ms.", flush=True)
        t_next = self.clock.time_of(k + 1)
        self.read_obs()
        new_chunk = self.file_finished(k)
        self.maybe_launch(k)
        cmd = self.agg.take(k)
        if cmd is None:
            self.hold(k)
        else:
            self.stats.votes_per_tick.append(cmd.n_votes)
            self.last_policy_action = cmd.policy_action
            self.command.update(cmd)
            self.command_arms(k)
        self.send_hands()
        self.print_status(k, cmd)
        self.records.append(self.record(k, cmd, new_chunk))
        self.stats.n_steps = k + 1
        precise_wait(t_next, time_func=time.monotonic)
        return True


def run_episode(predict, io: LoopIO, cfg: LoopConfig, *, records=None, stats=None):
    """Run one episode; returns ``(records, stats)``.

    ``predict(InferRequest) -> (H, action_dim)`` runs on the worker thread. ``records`` /
    ``stats`` may be passed in so the caller keeps them when the loop raises (Ctrl-C, a worker or
    hardware error). The caller stops the arms and resets the hands afterwards.
    """
    records = [] if records is None else records
    stats = EpisodeStats() if stats is None else stats
    ep = _Episode(predict, io, cfg, records, stats)
    try:
        with ep.keys:
            ep.worker.start()
            ep.start()
            for k in range(int(cfg.n_steps)):
                if not ep.tick(k):
                    break
    finally:
        stats.n_reanchors = 0 if ep.clock is None else ep.clock.n_reanchors
        shielded("stopping the inference worker", ep.worker.stop)
        print(f"[deploy] episode: {stats.summary()}", flush=True)
    return records, stats
