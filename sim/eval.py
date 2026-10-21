"""Evaluate a rotation-skill checkpoint in the training environment with deterministic actions.

    python sim/eval.py --headless --checkpoint logs/rotation/<run>/model_1199.pt
                       [--num-envs 64] [--steps 800] [--seed 42] [--nominal] [--video]
                       [--log-dir DIR]

Runs the actor's mean action for ``--steps`` control steps (800 = one 40 s episode) on the
randomized training plant, or with ``--nominal`` on the plant without domain randomization, and
writes ``<log-dir>/<run>_<checkpoint>[_nominal][_video].json`` (default log dir
``logs/rotation/eval``): the reward summed over the steps (mean over the envs), the handle speed in
RPM and the total number of early terminations. ``--video`` also renders an orbiting view of
environment 0 to an ``.mp4`` next to the JSON, and evaluates only that environment unless
``--num-envs`` is given. A failed run exits with code 1.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
from pathlib import Path

from isaaclab.app import AppLauncher

from rotation_env import ppo

parser = argparse.ArgumentParser(description="Evaluate a rotation-skill checkpoint.")
parser.add_argument(
    "--num-envs", type=int, default=None, help="parallel environments (default 64, 1 with --video)"
)
parser.add_argument("--steps", type=int, default=800, help="control steps (800 = one episode)")
parser.add_argument(
    "--seed", type=int, default=ppo.SEED, help="random seed of the randomized plant"
)
parser.add_argument("--nominal", action="store_true", help="no domain randomization")
parser.add_argument("--video", action="store_true", help="also render an environment to an mp4")
parser.add_argument(
    "--log-dir", default=os.path.join("logs", "rotation", "eval"), help="output directory"
)
AppLauncher.add_app_launcher_args(parser)
# Added after the AppLauncher options: add_app_launcher_args() parses the command line once, and a
# required option missing there would abort `--help`.
parser.add_argument("--checkpoint", required=True, help="RSL-RL checkpoint (model_<iteration>.pt)")
args_cli = parser.parse_args()
if args_cli.video:
    args_cli.enable_cameras = True
simulation_app = AppLauncher(args_cli).app

# Isaac Lab modules can be imported only after the app has started.
import torch

from dexport.skills.rotation.contract import NUM_JOINTS, OBS_DIM
from rotation_env.env import build_env_cfg, make_env

ORBIT_RADIUS = 0.5  # m, horizontal camera distance from the look-at point
CAMERA_HEIGHT = 0.35  # m above the env origin
LOOKAT_HEIGHT = 0.15  # m above the env origin


def evaluate(env, actor, steps: int, record: bool) -> tuple[dict, list]:
    """Roll out the mean action; return the metrics and, with ``record``, the rendered frames."""
    screw = env.scene["screw"]
    obs, _ = env.reset()
    origin = env.scene.env_origins[0].tolist()
    lookat = [origin[0], origin[1], origin[2] + LOOKAT_HEIGHT]
    if record:
        env.render()  # attaches the frame grabber, whose first frame is still empty
        for _ in range(5):  # let the renderer settle before the first frame
            env.sim.render()
    summed_reward = torch.zeros(env.num_envs, device=env.device)
    turned = torch.zeros(env.num_envs, device=env.device)
    angle = screw.data.joint_pos[:, 0].clone()
    early_terminations = 0
    frames = []
    for step in range(steps):
        if record:
            phi = 2.0 * math.pi * step / steps
            eye = [
                lookat[0] + ORBIT_RADIUS * math.cos(phi),
                lookat[1] + ORBIT_RADIUS * math.sin(phi),
                origin[2] + CAMERA_HEIGHT,
            ]
            env.sim.set_camera_view(eye, lookat)
        with torch.inference_mode():
            actions = actor(obs, stochastic_output=False)
        obs, reward, terminated, truncated, _ = env.step(actions)
        new_angle = screw.data.joint_pos[:, 0]
        # A reset env restarts at angle 0, so its last step is not counted.
        turned += torch.where(terminated | truncated, 0.0, new_angle - angle)
        angle = new_angle.clone()
        summed_reward += reward
        early_terminations += int(terminated.sum())
        if record:
            frame = env.render()
            if frame is not None and frame.ndim == 3 and frame.size > 0:
                frames.append(frame[..., :3].copy())
    seconds = steps * env.step_dt
    metrics = {
        "mean_summed_reward": float(summed_reward.mean()),
        "handle_rpm": float(turned.mean()) / (2.0 * math.pi) / (seconds / 60.0),
        "early_terminations_total": early_terminations,
    }
    return metrics, frames


def main() -> None:
    num_envs = args_cli.num_envs or (1 if args_cli.video else 64)
    cfg = build_env_cfg(
        num_envs, seed=args_cli.seed, device=args_cli.device, randomize=not args_cli.nominal
    )
    env = make_env(cfg, render_mode="rgb_array" if args_cli.video else None)
    try:
        actor = ppo.load_actor(args_cli.checkpoint, OBS_DIM, NUM_JOINTS, str(env.device))
        metrics, frames = evaluate(env, actor, args_cli.steps, record=args_cli.video)

        checkpoint = Path(args_cli.checkpoint).resolve()
        out_dir = Path(args_cli.log_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{checkpoint.parent.name}_{checkpoint.stem}"
        stem += ("_nominal" if args_cli.nominal else "") + ("_video" if args_cli.video else "")
        video = None
        if args_cli.video:
            if not frames:
                raise RuntimeError("no frames were rendered")
            import imageio.v2 as imageio

            video = out_dir / f"{stem}.mp4"
            imageio.mimsave(video, frames, fps=round(1.0 / env.step_dt), macro_block_size=None)
        result = {
            "checkpoint": str(checkpoint),
            "num_envs": num_envs,
            "steps": args_cli.steps,
            "seed": args_cli.seed,
            "nominal": args_cli.nominal,
            **metrics,
            "video": str(video) if video else None,
        }
        (out_dir / f"{stem}.json").write_text(json.dumps(result, indent=1) + "\n")
        print(f"[eval] {json.dumps(result)}", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        # simulation_app.close() ends the process with exit code 0, so a failure exits here.
        logging.getLogger(__name__).exception("evaluation failed")
        logging.shutdown()
        os._exit(1)
    simulation_app.close()
