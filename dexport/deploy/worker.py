"""The inference worker: one background thread that turns observation windows into flange targets.

The control loop submits an :class:`InferRequest` (the observation window and the chunk's base
flange pose, both snapshotted on the control thread) and polls for the :class:`InferResult`: the
raw chunk, its arm rows conjugated by the mount and composed onto the base pose, and its hand rows.

One thread, one request at a time: inference mutates the policy's noise scheduler, so concurrent
inferences would corrupt each other. A thread rather than a process suffices because torch
releases the GIL inside its kernels.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import numpy as np

from dexport.deploy.frames import chunk_to_flange_targets


@dataclass
class InferRequest:
    """One inference job. ``obs_window`` is already in the policy's frame; ``base_pos`` /
    ``base_quat`` (per arm) is the flange pose commanded at ``launch_step - 1``, onto which the
    chunk's relative arm actions are composed."""

    launch_step: int
    obs_window: list
    base_pos: dict
    base_quat: dict


@dataclass
class InferResult:
    """A finished chunk: ``chunk`` (H, action_dim) as predicted, and per arm the absolute flange
    targets ``arm_pos`` (H, 3), ``arm_quat`` (H, 4) xyzw and the hand targets ``hand`` (H, 20)."""

    launch_step: int
    chunk: np.ndarray
    arm_pos: dict
    arm_quat: dict
    hand: dict
    latency_s: float = 0.0
    latency_ticks: int = 0


class AsyncChunkWorker:
    """Runs ``predict(request) -> (H, action_dim) chunk`` on a background thread.

    ``arm_slices`` / ``hand_slices`` (the arm's and hand's columns of the chunk, ``slice``) and
    ``mounts`` are per arm. ``busy`` is True from ``submit`` until the result is published;
    ``idle_and_drained()`` additionally requires the result to have been polled, which is what
    the loop gates a launch on.
    """

    def __init__(self, predict, *, arms, arm_slices, hand_slices, mounts, control_dt):
        self.predict = predict
        self.arms = list(arms)
        self.arm_slices = dict(arm_slices)
        self.hand_slices = dict(hand_slices)
        self.mounts = dict(mounts)
        self.control_dt = float(control_dt)
        self._cv = threading.Condition()
        self._pending: InferRequest | None = None
        self._result: InferResult | None = None
        self._busy = False
        self._running = False
        self._thread: threading.Thread | None = None
        self._err: BaseException | None = None

    def start(self) -> "AsyncChunkWorker":
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="chunk-infer", daemon=True)
        self._thread.start()
        return self

    def stop(self, timeout: float = 30.0) -> None:
        """Stop after the inference in flight; warn if it does not finish within ``timeout``
        (a second worker would then race it on the policy)."""
        with self._cv:
            self._running = False
            self._cv.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                print(
                    f"[worker] WARNING: the inference worker is still running after "
                    f"{timeout:.0f} s; restart the process before the next episode.",
                    flush=True,
                )
            self._thread = None

    @property
    def busy(self) -> bool:
        with self._cv:
            return self._busy

    def idle_and_drained(self) -> bool:
        """No job in flight and no finished result waiting to be polled."""
        with self._cv:
            return (not self._busy) and self._result is None and self._pending is None

    def raise_if_failed(self) -> None:
        """Re-raise a worker-thread exception on the control thread."""
        if self._err is not None:
            raise RuntimeError(f"inference worker died: {self._err!r}") from self._err

    def submit(self, req: InferRequest) -> None:
        """Start ``req``. Only one request at a time: call when ``idle_and_drained()``."""
        with self._cv:
            if self._busy or self._pending is not None:
                raise RuntimeError("the inference worker is busy")
            self._pending = req
            self._busy = True
            self._cv.notify()

    def poll(self) -> InferResult | None:
        """The finished result, or None. Never blocks."""
        with self._cv:
            res, self._result = self._result, None
        return res

    def _loop(self) -> None:
        try:
            while True:
                with self._cv:
                    while self._running and self._pending is None:
                        self._cv.wait(timeout=0.25)
                    if not self._running:
                        return
                    req, self._pending = self._pending, None
                res = self._run_one(req)
                with self._cv:
                    self._result = res
                    self._busy = False
                    self._cv.notify_all()
        except BaseException as e:  # surfaced on the control thread
            self._err = e
            with self._cv:
                self._busy = False
                self._cv.notify_all()
            print(f"[worker] inference worker error: {e!r}", flush=True)

    def _run_one(self, req: InferRequest) -> InferResult:
        t0 = time.monotonic()
        chunk = np.asarray(self.predict(req), dtype=np.float64)
        arm_pos, arm_quat, hand = {}, {}, {}
        for a in self.arms:
            pos, quat = chunk_to_flange_targets(
                chunk[:, self.arm_slices[a]],
                np.asarray(req.base_pos[a], dtype=np.float64),
                np.asarray(req.base_quat[a], dtype=np.float64),
                self.mounts[a],
            )
            arm_pos[a] = np.asarray(pos, dtype=np.float64)
            arm_quat[a] = np.asarray(quat, dtype=np.float64)
            hand[a] = chunk[:, self.hand_slices[a]].astype(np.float64)
        latency_s = time.monotonic() - t0
        return InferResult(
            launch_step=req.launch_step,
            chunk=chunk,
            arm_pos=arm_pos,
            arm_quat=arm_quat,
            hand=hand,
            latency_s=latency_s,
            latency_ticks=int(np.ceil(latency_s / max(self.control_dt, 1e-9))),
        )
