"""One deploy session: a trained policy, or a recorded trajectory, on one or both Tianji arms with
Wuji hands, through the HATO loop (:mod:`dexport.deploy.loop`).

:func:`run` derives the arms, mounts, action layout, observation window and cameras from the
checkpoint (or the recording), opens the hardware and runs episodes: move the arms home, seat the
optional start grasp, show the cameras, run the loop, then stop the arms, open the hands and write
the episode log. A policy runs episode after episode ('r' between them) until ``episodes`` or
quit; a replay is one pass over its recording. Refused options raise
:class:`~dexport.deploy.errors.DeployError` before any hardware moves.
"""

from __future__ import annotations

import dataclasses
import functools
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from dexport.common.joints import MUJOCO_ORDER
from dexport.common.terminal import wait_for_key
from dexport.deploy.config import DeployConfig, hand_to_flange_t, mount_transform
from dexport.deploy.errors import DeployError
from dexport.deploy.frames import action_slices, to_policy_frame
from dexport.deploy.loop import ArmIO, EpisodeStats, LoopConfig, LoopIO, run_episode, shielded
from dexport.deploy.obs import obs_window
from dexport.hardware.tianji.arms import ARM_SIDE, SIDE_ARM
from dexport.skills.hand_models import HAND_MODELS

ARMS = {"A": ("A",), "B": ("B",), "both": ("A", "B")}
_HOME_PROMPT = "[deploy] Clear the workspace. Any key: move the arm(s) home; q: quit "
_NEXT_EPISODE_PROMPT = "[deploy] r: home the arm(s) and run another episode; any other key: quit "
#: Device index of a single-camera policy's camera when ``--camera`` does not give one.
DEFAULT_CAMERA_INDEX = 0


@dataclass
class Options:
    """What to run. Exactly one of ``ckpt`` (a run directory or checkpoint) and ``replay`` (a
    ``trajectories.zarr``) is set; the rig's settings are a :class:`DeployConfig`."""

    ckpt: str | None = None
    replay: str | None = None
    traj: str = "traj_0"
    arm: str | None = None  # "A" | "B" | "both"; a policy's arms come from its checkpoint
    hand: str = "wuji1"
    episodes: int | None = None  # policy only; None: until quit
    max_steps: int | None = None  # default ControlConfig.max_steps (policy), all frames (replay)
    diffusion_steps: int | None = None  # default ControlConfig.diffusion_steps
    spacer: bool = False  # the controller bracket's spacer is between flange and hand
    hand_init: str | None = None  # "screw": start each episode in the rotation skill's grasp
    cameras: dict = field(default_factory=dict)  # camera key -> device index
    log_dir: str | None = None


@dataclass
class Plan:
    """Everything resolved before any hardware is touched."""

    mode: str  # "policy" | "replay"
    arms: tuple
    hand: str
    mounts: dict
    predict: Callable
    build_obs_window: Callable
    loop: LoopConfig
    image_keys: tuple  # camera key of each policy image, in the policy's im_key order
    camera_index: dict  # distinct camera key -> device index
    hand_init: np.ndarray | None
    meta: dict


@dataclass
class Hardware:
    arms: object  # TianjiArms
    hands: dict  # arm -> WujiHand
    cameras: dict  # camera key -> DeployCamera


@dataclass
class Episode:
    records: list
    stats: EpisodeStats
    meta: dict


def load_policy(path: str):
    """The trained policy (a run directory's model_last.ckpt, or a checkpoint file), on the GPU
    when there is one."""
    import torch

    from dexport.minbc.inference import DiffusionAgent

    return DiffusionAgent.load(path, device="cuda:0" if torch.cuda.is_available() else "cpu")


def apply_sampler(agent, steps: int) -> None:
    """DDIM with ``steps`` reverse steps over the trained (epsilon-predicting) weights."""
    trained = f"{agent.config.dp.diffusion_method}/{int(agent.config.dp.diffusion_iters)}"
    try:
        agent.set_sampler("ddim", steps)
    except ValueError as e:
        raise DeployError(f"--diffusion-steps must be >= 1, got {steps}") from e
    print(f"[deploy] sampler: DDIM, {agent.num_inference_steps} steps (trained {trained})")


def _camera_index(image_keys, given: dict) -> dict:
    """Device index of each camera the policy uses (``--camera KEY=INDEX``; a single camera
    defaults to ``DEFAULT_CAMERA_INDEX``)."""
    keys = list(dict.fromkeys(image_keys))
    unknown = sorted(set(given) - set(keys))
    if unknown:
        raise DeployError(f"--camera {unknown}: the policy's cameras are {keys}")
    missing = [k for k in keys if k not in given]
    if missing and len(keys) > 1:
        raise DeployError(
            f"the policy uses the cameras {keys}: give each a device with --camera KEY=INDEX "
            f"(missing {missing})"
        )
    return {k: int(given.get(k, DEFAULT_CAMERA_INDEX)) for k in keys}


def _mounts(hand: str, arms, spacer: bool) -> dict:
    return {a: mount_transform(hand, ARM_SIDE[a], spacer) for a in arms}


def _standoff(spacer: bool) -> str:
    """The flange-to-palm standoff in force: a wrong bracket state is otherwise silent (the lever
    arm offsets every commanded rotation, and a pure translation not at all)."""
    mm = 1000.0 * hand_to_flange_t(spacer)[2]
    return f"flange-to-palm standoff {mm:.1f} mm ({'with' if spacer else 'no'} --spacer)"


def _loop_config(ctl, obs_horizon: int, n_steps: int, fail_on_hold: bool = False) -> LoopConfig:
    return LoopConfig(
        frequency=ctl.frequency,
        obs_horizon=obs_horizon,
        n_steps=n_steps,
        chunk_rows=ctl.chunk_rows,
        warmup_inferences=ctl.warmup_inferences,
        fail_on_hold=fail_on_hold,
    )


def _policy_arms(data) -> tuple:
    """The arms a checkpoint drives: both if it is bimanual, else the arm of its hand's side."""
    return ("A", "B") if data.bimanual else (SIDE_ARM[data.hand_side],)


def resolve(opts: Options, rig: DeployConfig) -> Plan:
    """The session's :class:`Plan` from the options and the checkpoint (or the recording);
    inconsistent options are refused here, before any hardware moves."""
    if (opts.ckpt is None) == (opts.replay is None):
        raise DeployError("pass exactly one of --ckpt and --replay")
    if opts.hand not in HAND_MODELS:
        raise DeployError(f"--hand must be one of {sorted(HAND_MODELS)}")
    for flag in ("episodes", "max_steps", "diffusion_steps"):
        value = getattr(opts, flag)
        if value is not None and value < 1:
            raise DeployError(f"--{flag.replace('_', '-')} must be >= 1, got {value}")
    if opts.replay is not None:
        return _resolve_replay(opts, rig.control)
    return _resolve_policy(opts, rig.control)


def _resolve_replay(opts: Options, ctl) -> Plan:
    from dexport.deploy.replay import RecordedChunks, load_recording

    if opts.episodes is not None:
        raise DeployError("--episodes does not apply to --replay (one pass over the recording)")
    if opts.arm is None:
        raise DeployError("--replay needs --arm A, B or both")
    if opts.hand_init is not None:
        raise DeployError("--hand-init applies to a policy")
    arms = ARMS[opts.arm]
    rec = load_recording(opts.replay, opts.traj, arms)
    n_steps = rec.n_frames - 1 if opts.max_steps is None else min(rec.n_frames - 1, opts.max_steps)
    print(
        f"[deploy] replay {opts.replay} {opts.traj}: {rec.n_frames} frames, arms "
        f"{'+'.join(arms)}, {opts.hand}; {_standoff(opts.spacer)}"
    )
    return Plan(
        mode="replay",
        arms=arms,
        hand=opts.hand,
        mounts=_mounts(opts.hand, arms, opts.spacer),
        predict=RecordedChunks(rec, arms),
        build_obs_window=lambda hist: [],
        loop=_loop_config(ctl, obs_horizon=1, n_steps=n_steps, fail_on_hold=True),
        image_keys=(),
        camera_index={},
        hand_init=None,
        meta={"replay": str(opts.replay), "traj": opts.traj, "n_frames": rec.n_frames},
    )


def _resolve_policy(opts: Options, ctl) -> Plan:
    from dexport.deploy.camera import camera_key

    agent = load_policy(opts.ckpt)
    apply_sampler(
        agent, ctl.diffusion_steps if opts.diffusion_steps is None else opts.diffusion_steps
    )
    data = agent.config.data
    arms = _policy_arms(data)
    if opts.arm is not None and ARMS[opts.arm] != arms:
        raise DeployError(
            f"the checkpoint drives arm(s) {'+'.join(arms)} ({data.hand_side} hand"
            f"{', bimanual' if data.bimanual else ''}), not --arm {opts.arm}"
        )
    hand_init = None
    if opts.hand_init is not None:
        from dexport.skills.rotation.contract import INIT_GRASP

        if not (HAND_MODELS[opts.hand].has_rotation_skill and arms == ("A",)):
            raise DeployError("--hand-init screw is the Wuji 1 left-hand start grasp (arm A)")
        hand_init = np.array([INIT_GRASP[n] for n in MUJOCO_ORDER], dtype=np.float64)
    if data.hand_action_source == "state":
        print("[deploy] the policy predicts measured hand states: they are sent as hand targets")
    mounts = _mounts(opts.hand, arms, opts.spacer)
    image_keys = ()
    if "img" in data.data_key:
        side = "left" if data.bimanual else data.hand_side
        image_keys = tuple(camera_key(k, side) for k in data.im_key)
    obs_horizon = agent.obs_horizon
    include_current = bool(agent.config.dp.obs_current_step)
    use_rel_ee = "rel_ee" in data.data_key
    mount_list = [mounts[a] for a in arms]

    def build_obs_window(hist):
        window = obs_window(hist, arms, obs_horizon, include_current, use_rel_ee)
        return to_policy_frame(window, mount_list)

    print(
        f"[deploy] policy {opts.ckpt}: arms {'+'.join(arms)}, {opts.hand}, cameras "
        f"{list(image_keys)}, obs horizon {obs_horizon}, {agent.total_horizon}-step chunks; "
        f"{_standoff(opts.spacer)}"
    )
    return Plan(
        mode="policy",
        arms=arms,
        hand=opts.hand,
        mounts=mounts,
        predict=lambda req: agent.predict_chunk(req.obs_window),
        build_obs_window=build_obs_window,
        loop=_loop_config(
            ctl, obs_horizon, ctl.max_steps if opts.max_steps is None else opts.max_steps
        ),
        image_keys=image_keys,
        camera_index=_camera_index(image_keys, opts.cameras),
        hand_init=hand_init,
        meta={
            "ckpt": str(opts.ckpt),
            "data_key": list(data.data_key),
            "im_key": list(data.im_key),
            "obs_horizon": obs_horizon,
            "total_horizon": agent.total_horizon,
            "diffusion_steps": int(agent.num_inference_steps),
        },
    )


def open_hardware(plan: Plan, rig: DeployConfig) -> Hardware:
    """Construct (not connect) the arms, hands and cameras."""
    from dexport.collect.config import CameraConfig
    from dexport.deploy.camera import DeployCamera
    from dexport.deploy.hand import WujiHand
    from dexport.hardware.tianji.arms import TianjiArms

    return Hardware(
        arms=TianjiArms(rig.tianji, plan.arms),
        hands={a: WujiHand(rig.hand, side=ARM_SIDE[a], hand=plan.hand) for a in plan.arms},
        cameras={k: DeployCamera(CameraConfig(k, i)) for k, i in plan.camera_index.items()},
    )


def _loop_io(plan: Plan, hw: Hardware) -> LoopIO:
    arm_slices, hand_slices = action_slices(len(plan.arms))
    arms = hw.arms

    def arm_io(i, a):
        return ArmIO(
            get_ee_pose=lambda: arms.get_ee_pose(a),
            solve_ik=lambda pos, quat: arms.solve_ik_deg(a, pos, quat),
            schedule_joints=lambda q, t: arms.schedule_joints_deg(a, q, t),
            arm_slice=arm_slices[i],
            hand_slice=hand_slices[i],
            mount=plan.mounts[a],
            hand=hw.hands[a],
        )

    images = None
    if plan.image_keys:
        cams = [hw.cameras[k] for k in plan.image_keys]

        def images():
            return np.stack([c.get_rgb() for c in cams])

    return LoopIO(
        arms={a: arm_io(i, a) for i, a in enumerate(plan.arms)},
        build_obs_window=plan.build_obs_window,
        stop_motion=arms.stop_motion,
        get_image_stack=images,
    )


def _save_log(log_dir, mode, k, episode: Episode) -> None:
    ts = time.strftime("%Y%m%d_%H%M%S", time.localtime(episode.meta["t_start"]))
    path = Path(log_dir) / f"{mode}_{ts}_ep{k}.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"meta": episode.meta, "records": episode.records}, f)
    print(f"[deploy] episode log ({len(episode.records)} steps) -> {path}", flush=True)


class _Devices:
    """The session's open devices: on exit, any hand that may be away from its open pose is opened,
    then every device is closed in reverse."""

    def __init__(self, hw: Hardware):
        self.hw = hw
        self.connected: list = []
        self.hands_away = False  # a hand may be away from its open pose

    def connect(self, arms) -> None:
        """Connect and home the arms (printing each home flange position), then connect the hands
        and cameras; a device counts as open as soon as its connect returns."""
        hw = self.hw
        hw.arms.connect()
        self.connected.append(hw.arms)
        hw.arms.home()
        for a in arms:
            pos = np.round(hw.arms.get_ee_pose(a)[0], 4).tolist()
            print(f"[deploy] arm {a} home flange position: {pos}", flush=True)
        self.hands_away = True
        for dev in (*hw.hands.values(), *hw.cameras.values()):
            dev.connect()
            self.connected.append(dev)

    def open_hands(self) -> None:
        for a, h in self.hw.hands.items():
            if h in self.connected:
                shielded(f"opening hand {a}", h.reset)
        self.hands_away = False

    def close(self) -> None:
        if self.hands_away:
            self.open_hands()
        for dev in reversed(self.connected):
            shielded("closing", dev.close)


def _confirm_home(plan: Plan, rig: DeployConfig) -> bool:
    """Show each arm's home joints and wait for a key; False if the operator quits."""
    for a in plan.arms:
        q = ", ".join(f"{v:+.1f}" for v in rig.tianji.home_deg[a])
        print(f"[deploy] arm {a} home joints (deg): [{q}]")
    return wait_for_key(_HOME_PROMPT) != "q"


def _run_one_episode(
    plan: Plan, io: LoopIO, devices: _Devices, opts: Options, k: int
) -> Episode | None:
    """Episode ``k``: seat the start grasp, show the cameras (None if the operator quits there),
    run the loop, then stop the arms, open the hands and write the episode log."""
    hw = devices.hw
    devices.hands_away = True
    if plan.hand_init is not None:
        for h in hw.hands.values():
            h.ramp_to(plan.hand_init)
    if hw.cameras:
        from dexport.hardware.camera import preview_cameras

        if not preview_cameras(hw.cameras.items()):
            return None
    ep = Episode([], EpisodeStats(), dict(plan.meta))
    ep.meta.update(
        mode=plan.mode,
        arms=list(plan.arms),
        hand=plan.hand,
        spacer=opts.spacer,
        mounts={a: m.tolist() for a, m in plan.mounts.items()},
        frequency=plan.loop.frequency,
        chunk_rows=plan.loop.chunk_rows,
        t_start=time.time(),
    )
    try:
        run_episode(plan.predict, io, plan.loop, records=ep.records, stats=ep.stats)
    finally:
        shielded("stopping the arms", hw.arms.stop_motion)
        devices.open_hands()
        ep.meta["stats"] = dataclasses.asdict(ep.stats)
        ep.meta["arms_summary"] = hw.arms.summary()
        print(f"[deploy] {ep.meta['arms_summary']}", flush=True)
        if opts.log_dir is not None:
            save = functools.partial(_save_log, opts.log_dir, plan.mode, k, ep)
            shielded("saving the episode log", save)
    return ep


def _another_episode(plan: Plan, opts: Options, n_done: int) -> bool:
    """A replay is one pass and ``episodes`` caps a policy run; within the cap the operator
    chooses ('r': another)."""
    if plan.mode == "replay" or (opts.episodes is not None and n_done >= opts.episodes):
        return False
    return wait_for_key(_NEXT_EPISODE_PROMPT) == "r"


def run(opts: Options, *, rig=None) -> list:
    """Run the session; returns its :class:`Episode` s. ``rig`` defaults to ``DeployConfig()``."""
    rig = DeployConfig() if rig is None else rig
    plan = resolve(opts, rig)
    hw = open_hardware(plan, rig)
    io = _loop_io(plan, hw)
    devices = _Devices(hw)
    episodes: list = []
    try:
        if not _confirm_home(plan, rig):
            return episodes
        devices.connect(plan.arms)
        while True:
            ep = _run_one_episode(plan, io, devices, opts, len(episodes))
            if ep is None:
                break
            episodes.append(ep)
            if not _another_episode(plan, opts, len(episodes)):
                break
            hw.arms.home()
    finally:
        devices.close()
    return episodes
