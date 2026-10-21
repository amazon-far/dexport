"""The Tianji (Marvin CCS-M6) arms: one driver for one arm or both.

The Marvin SDK (``tj_marvin``, imported at connect) is single-client, so one ``MarvinRobot`` and one
servo thread drive every active arm, with one ``command_joints`` call per tick. The SDK hands
joint set-points (degrees) to the arm's joint impedance controller; the servo thread feeds it a
velocity- and acceleration-limited reference built from timed joint waypoints
(:class:`JointWaypointQueue`). A command watchdog brakes an arm to a stop once its queued plan
has run out and nothing has been scheduled for ``command_timeout_s``; homing moves disarm it.
FK and IK use PyRoki (:mod:`dexport.hardware.tianji.pyroki_ik`) on the two-arm URDF, in radians;
degrees are converted at the SDK boundary. Arm "A" is the left arm, "B" the right (``ARM_SIDE``).
"""

from __future__ import annotations

import dataclasses
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from dexport.common.pose import mat_to_pos_quat, pos_quat_to_mat

#: Arm name -> the side of its hand, and back.
ARM_SIDE = {"A": "left", "B": "right"}
SIDE_ARM = {side: arm for arm, side in ARM_SIDE.items()}
#: Arm name -> the suffix of its links and joints in the two-arm URDF (``Joint1_L``, ``flange_R``).
ARM_URDF_TAG = {"A": "L", "B": "R"}

#: Marvin CCS-M6 joint limits, degrees (J1..J7).
JOINT_LIMITS_DEG = (
    (-178, 178),
    (-120, 120),
    (-178, 178),
    (-145, 60),
    (-178, 178),
    (-60, 60),
    (-90, 90),
)


@dataclass
class TianjiConfig:
    """How the driver talks to ``tj_marvin.MarvinRobot``. ``K`` / ``D`` replace the marvin YAML's
    joint impedance; ``tool_payload`` is merged over its ``tool`` entry."""

    marvin_config: str | None = None  # default $TJ_MARVIN_ROOT/configs/marvin_config.yml
    urdf_path: str | None = None  # the two-arm URDF; default $TIANJI_URDF
    robot_ip: str | None = None  # default: the marvin YAML's robot_ip
    #: Per-arm gravity-compensation tool folder (with the Wuji hand mounted), relative to the
    #: marvin YAML's directory: where tj_marvin's dynamics identification (hw_test/dyn_id)
    #: writes it.
    tool_payload: dict | None = field(
        default_factory=lambda: {
            "A": "hw_test/dyn_id/output/left_arm_A_hand_wuji",
            "B": "hw_test/dyn_id/output/right_arm_B_hand_wuji",
        }
    )
    K: list = field(default_factory=lambda: [8.0, 8.0, 8.0, 6.0, 6.0, 6.0, 6.0])  # N·m/deg
    D: list = field(default_factory=lambda: [0.48] * 7)  # N·m/(deg/s)
    servo_hz: float = 100.0
    max_joint_vel_deg_s: float = 180.0
    max_joint_acc_deg_s2: float = 3000.0
    home_max_joint_vel_deg_s: float = 30.0  # homing moves are slower than playback
    home_max_joint_acc_deg_s2: float = 300.0
    command_timeout_s: float = 1.5
    home_move_timeout_s: float = 15.0
    home_deg: dict = field(
        default_factory=lambda: {
            "A": [-14.0, -25.0, 45.0, -110.0, 101.0, 31.0, -45.0],
            "B": [14.0, -25.0, -45.0, -110.0, -101.0, 31.0, 45.0],
        }
    )
    # PyRoki IK (Levenberg-Marquardt, warm-started at the previous solution)
    ik_pos_weight: float = 50.0
    ik_ori_weight: float = 10.0
    ik_rest_weight: float = 0.05  # pull towards the warm start
    ik_limit_weight: float = 100.0
    ik_lambda_initial: float = 1.0
    ik_pos_tol_mm: float = 1.0  # a solve counts as converged within these
    ik_rot_tol_deg: float = 1.0


def add_tianji_args(parser) -> None:
    """The rig flags of every tool that drives the arms: ``--urdf``, ``--marvin-config``,
    ``--robot-ip`` (see :func:`tianji_config_from_args`)."""
    parser.add_argument("--urdf", help="two-arm URDF (default $TIANJI_URDF)")
    parser.add_argument(
        "--marvin-config", help="marvin YAML (default $TJ_MARVIN_ROOT/configs/marvin_config.yml)"
    )
    parser.add_argument("--robot-ip", help="override the marvin YAML's robot_ip")


def tianji_config_from_args(args, cfg: TianjiConfig | None = None) -> TianjiConfig:
    """``cfg`` (default ``TianjiConfig()``) with the flags of :func:`add_tianji_args` that were
    given."""
    flags = {"urdf_path": args.urdf, "marvin_config": args.marvin_config, "robot_ip": args.robot_ip}
    given = {k: v for k, v in flags.items() if v is not None}
    return dataclasses.replace(TianjiConfig() if cfg is None else cfg, **given)


def resolve_arm_urdf(urdf_path: str | None) -> str:
    """The two-arm URDF: ``urdf_path`` if given, else ``$TIANJI_URDF``."""
    path = urdf_path or os.environ.get("TIANJI_URDF")
    if not path or not Path(path).expanduser().exists():
        raise FileNotFoundError(
            f"two-arm URDF not found ({path!r}): pass --urdf or set TIANJI_URDF to the URDF file"
        )
    return str(Path(path).expanduser().resolve())


def require_tj_marvin(marvin_config: str | None) -> Path:
    """Import check for the Marvin SDK; returns the marvin YAML (default
    ``$TJ_MARVIN_ROOT/configs/marvin_config.yml``)."""
    try:
        import tj_marvin  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "the Tianji arms need tj_marvin (the Marvin SDK): install it into this environment "
            "or put its checkout on PYTHONPATH"
        ) from exc
    if not marvin_config:
        root = os.environ.get("TJ_MARVIN_ROOT", "").strip()
        if not root:
            raise FileNotFoundError(
                "no marvin YAML: pass --marvin-config or set TJ_MARVIN_ROOT to the tj_marvin "
                "checkout"
            )
        marvin_config = str(Path(root) / "configs" / "marvin_config.yml")
    return Path(marvin_config).expanduser().resolve()


class JointWaypointQueue:
    """Timed joint waypoints (degrees, monotonic seconds) followed under velocity and acceleration
    limits.

    The reference is the piecewise-linear interpolation of the waypoints at the current time,
    held at the last one once the plan runs out. A waypoint whose arrival time has passed is
    dropped; one too far to reach in time arrives later (at ``vmax``) rather than faster. The
    commanded joints follow the reference under an acceleration clamp, braking towards the plan's
    final waypoint, so the reference's velocity steps never become an overshoot.
    """

    def __init__(self, q0, *, vmax, amax, dt, lo, hi):
        self.cmd = np.asarray(q0, dtype=np.float64).copy()
        self.vel = np.zeros(self.cmd.shape[0], dtype=np.float64)
        self.vmax, self.amax, self.dt = float(vmax), float(amax), float(dt)
        self.lo = np.asarray(lo, dtype=np.float64)
        self.hi = np.asarray(hi, dtype=np.float64)
        self._times: list[float] = [time.monotonic()]
        self._qs: list[np.ndarray] = [self.cmd.copy()]

    @property
    def final_target(self) -> np.ndarray:
        return self._qs[-1]

    @property
    def plan_end_time(self) -> float:
        """When the queued plan runs out (-inf when nothing is queued)."""
        return self._times[-1] if len(self._times) > 1 else -np.inf

    def _clip(self, q) -> np.ndarray:
        return np.clip(np.asarray(q, dtype=np.float64).reshape(self.cmd.shape), self.lo, self.hi)

    def _advance(self, desired_v):
        self.vel = np.clip(
            desired_v, self.vel - self.amax * self.dt, self.vel + self.amax * self.dt
        )
        self.cmd = np.clip(self.cmd + self.vel * self.dt, self.lo, self.hi)
        return self.cmd, self.vel

    def _min_duration(self, q_from, q_to) -> float:
        return float(np.max(np.abs(np.asarray(q_to) - np.asarray(q_from)))) / self.vmax

    def brake_to_stop(self):
        """One tick of decelerating to rest at ``amax``."""
        return self._advance(np.zeros_like(self.vel))

    def enqueue(self, q_deg, t_arrive: float) -> bool:
        """Append a waypoint; False (dropped) if ``t_arrive`` has already passed."""
        if t_arrive <= time.monotonic():
            return False
        q = self._clip(q_deg)
        t_prev, q_prev = self._times[-1], self._qs[-1]
        self._times.append(max(t_arrive, t_prev + max(self._min_duration(q_prev, q), 1e-4)))
        self._qs.append(q)
        return True

    def set_target(self, q_deg) -> None:
        """Drop the plan and go to ``q_deg`` as fast as the limits allow."""
        t_now = time.monotonic()
        q = self._clip(q_deg)
        self._times = [t_now, t_now + max(self._min_duration(self.cmd, q), 1e-3)]
        self._qs = [self.cmd.copy(), q]

    def clear_pending(self) -> None:
        """Drop all future motion; hold where the reference is."""
        self._times = [time.monotonic()]
        self._qs = [self.cmd.copy()]

    def step(self, t_now: float):
        """One servo tick: advance the commanded joints towards the reference at ``t_now``."""
        while len(self._times) >= 2 and self._times[1] <= t_now:
            self._times.pop(0)
            self._qs.pop(0)
        if len(self._times) == 1:
            ref = self._qs[0]
            self._times[0] = t_now
        else:
            t0, t1 = self._times[0], self._times[1]
            alpha = 0.0 if t1 <= t0 else float(min(max((t_now - t0) / (t1 - t0), 0.0), 1.0))
            ref = self._qs[0] + alpha * (self._qs[1] - self._qs[0])
        v_des = (ref - self.cmd) / self.dt
        v_cap = np.minimum(
            self.vmax, np.sqrt(2.0 * self.amax * np.abs(self.final_target - self.cmd))
        )
        return self._advance(np.sign(v_des) * np.minimum(np.abs(v_des), v_cap))


@dataclass
class _Arm:
    queue: JointWaypointQueue
    ik: object
    seed: np.ndarray  # IK warm start, radians
    measured_deg: np.ndarray  # measured joints as of the last servo tick
    watchdog_armed: bool = False
    watchdog_tripped: bool = False
    last_cmd_t: float = 0.0


class TianjiArms:
    """Lifecycle, measured flange poses, IK and joint-waypoint streaming for ``arms`` ("A" and/or
    "B"). Every method takes the arm name; arrival times are ``time.monotonic()`` seconds."""

    def __init__(self, cfg: TianjiConfig, arms):
        self.cfg = cfg
        self.arms = tuple(arms)
        if not self.arms or not set(self.arms) <= set(ARM_SIDE):
            raise ValueError(f"arms must be 'A' and/or 'B', got {arms!r}")
        lims = np.asarray(JOINT_LIMITS_DEG, dtype=np.float64)
        self.joint_lo, self.joint_hi = lims[:, 0], lims[:, 1]
        self._robot = None
        self._state: dict[str, _Arm] = {}
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._running = False
        self._servo_err: BaseException | None = None
        self.n_ik = self.n_ik_unconverged = self.n_ik_clipped = 0
        self.n_dropped_waypoints = 0
        self.n_watchdog_trips = 0

    # lifecycle
    def connect(self) -> "TianjiArms":
        """Build the IK, connect and configure the SDK, start the servo thread. The IK is built
        first, so a missing PyRoki stack fails before the arms are touched; if anything after
        the SDK connects fails, the SDK is disconnected again."""
        cfg = self.cfg
        urdf = resolve_arm_urdf(cfg.urdf_path)
        yaml_path = require_tj_marvin(cfg.marvin_config)
        from tj_marvin import MarvinRobot

        from dexport.hardware.tianji.pyroki_ik import PyrokiArmIk

        mcfg = self._marvin_config(yaml_path)
        iks = {}
        for a in self.arms:
            print(f"[tianji] building the IK of arm {a} (one-time JIT compile)...", flush=True)
            iks[a] = PyrokiArmIk(urdf, a, np.deg2rad(np.asarray(cfg.home_deg[a], float)), cfg)
        print(
            f"[tianji] connecting arms {list(self.arms)} at {mcfg.get('robot_ip')}: "
            f"K={mcfg['joint_impedance']['K']} D={mcfg['joint_impedance']['D']}",
            flush=True,
        )
        try:
            self._robot = MarvinRobot(mcfg, base_dir=yaml_path.parent)
            self._robot.connect()
            try:
                self._robot.configure(arms=list(self.arms))
            except Exception as exc:
                tool = mcfg.get("tool") or {}
                raise RuntimeError(
                    f"MarvinRobot.configure() failed ({exc!r}). The arms' tool payload folders "
                    f"{ {a: tool.get(a) for a in self.arms} } (relative to {yaml_path.parent}) "
                    "are written by tj_marvin's dynamics identification; identify the payload "
                    "first or set TianjiConfig.tool_payload."
                ) from exc
            for a in self.arms:
                q0 = np.asarray(self._robot.get_joint_pos(a), dtype=np.float64).reshape(7)
                queue = JointWaypointQueue(
                    q0,
                    vmax=cfg.max_joint_vel_deg_s,
                    amax=cfg.max_joint_acc_deg_s2,
                    dt=1.0 / float(cfg.servo_hz),
                    lo=self.joint_lo,
                    hi=self.joint_hi,
                )
                self._state[a] = _Arm(
                    queue=queue, ik=iks[a], seed=np.deg2rad(q0), measured_deg=q0.copy()
                )
            self._running = True
            self._thread = threading.Thread(
                target=self._servo_loop, name="tianji-servo", daemon=True
            )
            self._thread.start()
            time.sleep(0.1)  # one servo tick before anyone reads the feedback
        except BaseException:
            self.close()
            raise
        return self

    def _marvin_config(self, yaml_path: Path) -> dict:
        """The marvin YAML with this driver's arms, robot IP, joint impedance and tool payload;
        warns about a configured tool payload folder that does not exist."""
        import yaml

        cfg = self.cfg
        with open(yaml_path) as f:
            mcfg = yaml.safe_load(f)
        mcfg["active_arms"] = list(self.arms)
        if cfg.robot_ip is not None:
            mcfg["robot_ip"] = cfg.robot_ip
        mcfg["joint_impedance"] = {
            **(mcfg.get("joint_impedance") or {}),
            "K": [float(v) for v in cfg.K],
            "D": [float(v) for v in cfg.D],
        }
        if cfg.tool_payload is not None:
            mcfg["tool"] = {**(mcfg.get("tool") or {}), **cfg.tool_payload}
        tool = mcfg.get("tool") or {}
        for a in self.arms:
            if isinstance(tool.get(a), str) and not (yaml_path.parent / tool[a]).exists():
                print(
                    f"[tianji] WARNING: arm {a}'s tool payload folder {yaml_path.parent / tool[a]}"
                    " does not exist",
                    flush=True,
                )
        return mcfg

    def close(self) -> None:
        """Stop the servo thread and disconnect the SDK (safe to call at any point)."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._robot is not None:
            try:
                self._robot.disconnect()
            finally:
                self._robot = None

    def __enter__(self) -> "TianjiArms":
        return self.connect()

    def __exit__(self, *exc) -> None:
        self.close()

    # state, FK, IK
    def joints_deg(self, arm: str) -> np.ndarray:
        """Measured joints (degrees), as of the last servo tick."""
        with self._lock:
            return self._state[arm].measured_deg.copy()

    def fk(self, arm: str, q_deg) -> tuple[np.ndarray, np.ndarray]:
        """Flange (pos, quat_xyzw) in the robot base frame for joints ``q_deg``."""
        T = self._state[arm].ik.forward_kinematics(np.deg2rad(np.asarray(q_deg, np.float64)))
        return mat_to_pos_quat(T)

    def get_ee_pose(self, arm: str) -> tuple[np.ndarray, np.ndarray]:
        """Measured flange pose (pos, quat_xyzw)."""
        self._raise_if_servo_failed()
        return self.fk(arm, self.joints_deg(arm))

    def solve_ik_deg(self, arm: str, pos, quat_xyzw) -> np.ndarray:
        """Joints (degrees, clipped to the limits) reaching the flange pose, warm-started at the
        previous solution; the warm start becomes this (unclipped) solution."""
        self._raise_if_servo_failed()
        s = self._state[arm]
        target = pos_quat_to_mat(np.asarray(pos, np.float64), np.asarray(quat_xyzw, np.float64))
        q_rad, converged, _, _ = s.ik.solve(target, s.seed)
        s.seed = np.asarray(q_rad, dtype=np.float64).copy()
        q_raw = np.rad2deg(s.seed)
        q_deg = np.clip(q_raw, self.joint_lo, self.joint_hi)
        self.n_ik += 1
        self.n_ik_unconverged += not converged
        self.n_ik_clipped += bool(np.any(np.abs(q_raw - q_deg) > 1e-9))
        return q_deg

    # commands
    def schedule_joints_deg(self, arm: str, q_deg, t_arrive: float) -> bool:
        """Queue a joint waypoint to be reached at monotonic time ``t_arrive`` (arms the
        watchdog). False if it was dropped because ``t_arrive`` had already passed."""
        self._raise_if_servo_failed()
        q = np.asarray(q_deg, dtype=np.float64).reshape(7)
        with self._lock:
            s = self._state[arm]
            ok = s.queue.enqueue(q, float(t_arrive))
            s.watchdog_armed, s.watchdog_tripped = True, False
            s.last_cmd_t = time.monotonic()
        if not ok:
            self.n_dropped_waypoints += 1
            print(
                f"[tianji] WARNING: arm {arm} dropped a waypoint scheduled in the past.", flush=True
            )
        return ok

    def set_target_joints_deg(self, arm: str, q_deg) -> None:
        """Drop the plan and move to ``q_deg`` (non-blocking; disarms the watchdog)."""
        with self._lock:
            s = self._state[arm]
            s.queue.set_target(q_deg)
            s.watchdog_armed = s.watchdog_tripped = False

    def stop_motion(self, arm: str | None = None) -> None:
        """Drop queued motion and hold (all arms by default)."""
        with self._lock:
            for a in self.arms if arm is None else (arm,):
                s = self._state[a]
                s.queue.clear_pending()
                s.watchdog_armed = s.watchdog_tripped = False

    def reset_to_joints(self, q_by_arm: dict, timeout: float | None = None, tol_deg=0.75) -> bool:
        """Blocking move of the given arms to ``q_by_arm`` (degrees) at the homing speed limits.

        Done once every arm's commanded and measured joints are within ``tol_deg`` of the target
        and its commanded speed is below 1 deg/s; returns False on timeout. Either way the IK warm
        starts are re-seeded at the measured joints."""
        self._raise_if_servo_failed()
        timeout = self.cfg.home_move_timeout_s if timeout is None else float(timeout)
        saved = {}
        with self._lock:
            for a in q_by_arm:
                qu = self._state[a].queue
                saved[a] = (qu.vmax, qu.amax)
                qu.vmax = min(qu.vmax, float(self.cfg.home_max_joint_vel_deg_s))
                qu.amax = min(qu.amax, float(self.cfg.home_max_joint_acc_deg_s2))
        try:
            for a, q in q_by_arm.items():
                self.set_target_joints_deg(a, q)
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                self._raise_if_servo_failed()
                done, measured = True, {}
                with self._lock:
                    for a in q_by_arm:
                        s = self._state[a]
                        tgt, measured[a] = s.queue.final_target.copy(), s.measured_deg.copy()
                        done &= bool(
                            np.max(np.abs(s.queue.cmd - tgt)) < tol_deg
                            and np.max(np.abs(s.queue.vel)) < 1.0
                            and np.max(np.abs(measured[a] - tgt)) < tol_deg
                        )
                if done:
                    for a in q_by_arm:
                        self._state[a].seed = np.deg2rad(measured[a])
                    return True
                time.sleep(0.02)
            print("[tianji] WARNING: the move to the home joints timed out.", flush=True)
            with self._lock:
                for a in q_by_arm:
                    self._state[a].seed = np.deg2rad(self._state[a].measured_deg.copy())
            return False
        finally:
            with self._lock:
                for a, (v, acc) in saved.items():
                    self._state[a].queue.vmax, self._state[a].queue.amax = v, acc

    def home(self) -> bool:
        """``reset_to_joints`` to the configured home of every arm."""
        return self.reset_to_joints({a: self.cfg.home_deg[a] for a in self.arms})

    def summary(self) -> str:
        return (
            f"IK: {self.n_ik} solves, {self.n_ik_unconverged} not converged, {self.n_ik_clipped} "
            f"clipped to the joint limits; {self.n_dropped_waypoints} waypoints dropped; "
            f"watchdog tripped {self.n_watchdog_trips}x"
        )

    # servo thread (the only SDK caller once connected)
    def _servo_tick(self, t_now: float) -> None:
        timeout = float(self.cfg.command_timeout_s)
        cmds, tripped = {}, []
        with self._lock:
            for a in self.arms:
                s = self._state[a]
                fresh_until = max(s.last_cmd_t, s.queue.plan_end_time)
                if s.watchdog_armed and timeout > 0.0 and t_now - fresh_until > timeout:
                    if not s.watchdog_tripped:
                        s.watchdog_tripped = True
                        s.queue.clear_pending()
                        tripped.append(a)
                    cmd, _ = s.queue.brake_to_stop()
                else:
                    cmd, _ = s.queue.step(t_now)
                cmds[a] = cmd.copy()
        for a in tripped:
            self.n_watchdog_trips += 1
            print(
                f"[tianji] WARNING: nothing scheduled for arm {a} for {timeout:.1f} s: braking "
                "to a stop.",
                flush=True,
            )
        self._robot.command_joints(**{f"joints_{a}": cmds[a].tolist() for a in self.arms})
        fb = {a: np.asarray(self._robot.get_joint_pos(a), np.float64).reshape(7) for a in self.arms}
        with self._lock:
            for a in self.arms:
                self._state[a].measured_deg = fb[a]

    def _servo_loop(self) -> None:
        dt = 1.0 / float(self.cfg.servo_hz)
        try:
            next_t = time.monotonic()
            while self._running:
                self._servo_tick(time.monotonic())
                next_t += dt
                pause = next_t - time.monotonic()
                if pause > 0:
                    time.sleep(pause)
                else:
                    next_t = time.monotonic()  # fell behind: do not try to catch up
        except BaseException as e:  # surfaced on the caller's thread
            self._servo_err = e
            self._running = False
            print(f"[tianji] servo thread error: {e!r}", flush=True)

    def _raise_if_servo_failed(self) -> None:
        if self._servo_err is not None:
            raise RuntimeError(
                f"the Tianji servo thread died: {self._servo_err!r}"
            ) from self._servo_err
