"""Deploy a trained policy, or replay a recording, on the Tianji arms with Wuji hands.

    python -m dexport.deploy --ckpt RUN_DIR [--hand wuji2] [--camera KEY=INDEX ...]
    python -m dexport.deploy --replay SESSION/trajectories.zarr --traj traj_3 --arm B [--hand wuji2]

e.g. pick-and-place / screwdriver (Wuji 1): ``--ckpt RUN`` (``--hand-init screw`` seats the
screwdriver grasp); drawer: ``--ckpt RUN --hand wuji2 --camera right_rgb=0 --camera left_rgb=2``;
scan (both arms): ``--ckpt RUN --hand wuji2 --camera left_rgb=0 --camera right_rgb=2``. The arms
come from the checkpoint (bimanual: both; else its hand side, left = arm A, right = arm B). Replay a
recording, with the ``--hand`` it was collected with, before deploying a policy trained on it: the
arm then retraces the recorded controller motion through the calibrated mount. In this terminal,
Space pauses and Esc ends an episode.
"""

from __future__ import annotations

import argparse
import sys

from dexport.deploy.config import ControlConfig, DeployConfig
from dexport.deploy.errors import DeployError
from dexport.deploy.session import Options, run
from dexport.hardware.tianji.arms import ARM_SIDE, add_tianji_args, tianji_config_from_args


def _camera(text: str) -> tuple[str, int]:
    key, sep, index = text.partition("=")
    if not sep or not index.strip().isdigit():
        raise argparse.ArgumentTypeError(f"expected KEY=INDEX (e.g. right_rgb=0), got {text!r}")
    return key.strip(), int(index)


def _positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected an integer, got {text!r}") from None
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {value}")
    return value


def parse_args(argv=None) -> argparse.Namespace:
    ctl = ControlConfig()
    ap = argparse.ArgumentParser(
        prog="python -m dexport.deploy",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--ckpt", help="run directory (its model_last.ckpt) or checkpoint file")
    src.add_argument("--replay", help="trajectories.zarr of a recorded session")
    ap.add_argument("--traj", default=Options.traj, help=f"trajectory to replay ({Options.traj})")
    ap.add_argument(
        "--arm", choices=["A", "B", "both"], help="arm(s): A = left, B = right (needed by --replay)"
    )
    ap.add_argument(
        "--hand",
        choices=["wuji1", "wuji2"],
        default=Options.hand,
        help=f"hand model (wuji1: USB, wuji2: Ethernet; default {Options.hand})",
    )
    ap.add_argument(
        "--episodes",
        type=_positive_int,
        help="number of policy episodes (default: until quit)",
    )
    ap.add_argument(
        "--max-steps",
        type=_positive_int,
        help=f"steps per episode (policy default {ctl.max_steps}, replay: all)",
    )
    ap.add_argument(
        "--diffusion-steps", type=_positive_int, help=f"DDIM steps (default {ctl.diffusion_steps})"
    )
    ap.add_argument(
        "--spacer",
        action="store_true",
        help="the controller bracket's 12 mm spacer sits between flange and hand",
    )
    ap.add_argument(
        "--hand-init",
        choices=["screw"],
        help="start each episode in the rotation skill's grasp (Wuji 1, arm A)",
    )
    ap.add_argument(
        "--camera",
        type=_camera,
        action="append",
        default=[],
        metavar="KEY=INDEX",
        help="device index of a policy camera (repeat per camera key)",
    )
    ap.add_argument(
        "--home",
        nargs=8,
        action="append",
        default=[],
        metavar=("ARM", "J1", "J2", "J3", "J4", "J5", "J6", "J7"),
        help="home joint angles of an arm, degrees",
    )
    add_tianji_args(ap)
    ap.add_argument("--log-dir", help="write one pickle log per episode here")
    return ap.parse_args(argv)


def options_from_args(args: argparse.Namespace) -> Options:
    return Options(
        ckpt=args.ckpt,
        replay=args.replay,
        traj=args.traj,
        arm=args.arm,
        hand=args.hand,
        episodes=args.episodes,
        max_steps=args.max_steps,
        diffusion_steps=args.diffusion_steps,
        spacer=args.spacer,
        hand_init=args.hand_init,
        cameras=dict(args.camera),
        log_dir=args.log_dir,
    )


def rig_from_args(args: argparse.Namespace) -> DeployConfig:
    """The default rig with the command line's Tianji settings (``--urdf``, ``--marvin-config``,
    ``--robot-ip``, ``--home``)."""
    tianji = tianji_config_from_args(args)
    for arm, *q in args.home:
        if arm not in ARM_SIDE:
            raise SystemExit(f"--home: arm must be A or B, got {arm!r}")
        try:
            tianji.home_deg = {**tianji.home_deg, arm: [float(v) for v in q]}
        except ValueError as e:
            raise SystemExit(f"--home {arm}: joint angles must be numbers") from e
    return DeployConfig(tianji=tianji)


def main(argv=None) -> int:
    args = parse_args(argv)
    opts, rig = options_from_args(args), rig_from_args(args)
    try:
        run(opts, rig=rig)
    except DeployError as e:
        raise SystemExit(f"[deploy] {e}") from None
    except KeyboardInterrupt:
        print("\n[deploy] Ctrl-C: session ended.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
