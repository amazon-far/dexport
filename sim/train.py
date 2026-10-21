"""Train the in-hand rotation skill with PPO in Isaac Lab.

    python sim/train.py --headless [--num-envs 4096] [--max-iterations 1200] [--seed 42]
                        [--log-dir DIR] [--wandb] [--device cuda:0]

The defaults are the full training run. RSL-RL saves ``model_<iteration>.pt`` every 100
iterations and at the end, plus TensorBoard logs, to ``--log-dir`` (default
``logs/rotation/<timestamp>`` under the current directory). ``--wandb`` also logs to Weights &
Biases (entity: ``$WANDB_USERNAME``). A failed run exits with code 1.
"""

from __future__ import annotations

import argparse
import logging
import os
from datetime import datetime

from isaaclab.app import AppLauncher

from rotation_env import ppo

parser = argparse.ArgumentParser(description="Train the in-hand rotation skill with PPO.")
parser.add_argument("--num-envs", type=int, default=ppo.NUM_ENVS, help="parallel environments")
parser.add_argument("--max-iterations", type=int, default=ppo.MAX_ITERATIONS, help="PPO iterations")
parser.add_argument(
    "--seed",
    type=int,
    default=ppo.SEED,
    help="random seed (Isaac Lab seeds Python, numpy and torch with it)",
)
parser.add_argument(
    "--log-dir", default=None, help="output directory (default: logs/rotation/<timestamp>)"
)
parser.add_argument("--wandb", action="store_true", help="also log to Weights & Biases")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
simulation_app = AppLauncher(args_cli).app

# Isaac Lab modules can be imported only after the app has started.
from rotation_env.env import build_env_cfg, make_env


def run_ppo(env, args: argparse.Namespace) -> None:
    """Train the actor-critic on ``env`` with RSL-RL PPO."""
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.runners import OnPolicyRunner

    log_dir = args.log_dir or os.path.join(
        "logs", "rotation", datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    os.makedirs(log_dir, exist_ok=True)
    print(f"[train] logging to {os.path.abspath(log_dir)}", flush=True)
    cfg = ppo.train_cfg(wandb=args.wandb)
    runner = OnPolicyRunner(RslRlVecEnvWrapper(env), cfg, log_dir=log_dir, device=str(env.device))
    runner.learn(
        num_learning_iterations=args.max_iterations,
        init_at_random_ep_len=ppo.INIT_AT_RANDOM_EP_LEN,
    )


def main() -> None:
    env = make_env(build_env_cfg(args_cli.num_envs, args_cli.seed, args_cli.device))
    try:
        run_ppo(env, args_cli)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        # simulation_app.close() ends the process with exit code 0, so a failure exits here.
        logging.getLogger(__name__).exception("training failed")
        logging.shutdown()
        os._exit(1)
    simulation_app.close()
