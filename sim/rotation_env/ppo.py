"""PPO settings of the in-hand rotation skill and the actor loader.

RSL-RL ``OnPolicyRunner``: 4096 environments x 24-step rollouts x 1200 iterations; actor and critic
MLPs [512, 256, 128] with ELU and running observation normalization; a Gaussian policy with initial
std 1.0; PPO with an adaptive learning rate. The actor reads the ``policy`` observation group, the
critic the privileged ``critic`` group. Importable without Isaac Lab (torch and rsl_rl are imported
by :func:`load_actor` only).
"""

from __future__ import annotations

import copy

NUM_ENVS = 4096
MAX_ITERATIONS = 1200
SEED = 42
NUM_STEPS_PER_ENV = 24
SAVE_INTERVAL = 100
INIT_AT_RANDOM_EP_LEN = True  # start each env at a random step of its first episode
WANDB_PROJECT = "dexport-rotation"

ACTOR = {
    "class_name": "MLPModel",
    "obs_normalization": True,
    "hidden_dims": [512, 256, 128],
    "activation": "elu",
    "distribution_cfg": {"class_name": "GaussianDistribution", "init_std": 1.0},
}
CRITIC = {
    "class_name": "MLPModel",
    "obs_normalization": True,
    "hidden_dims": [512, 256, 128],
    "activation": "elu",
}
ALGORITHM = {
    "class_name": "PPO",
    "value_loss_coef": 1.0,
    "use_clipped_value_loss": True,
    "clip_param": 0.2,
    "entropy_coef": 0.0,
    "num_learning_epochs": 5,
    "num_mini_batches": 4,
    "learning_rate": 5e-3,
    "schedule": "adaptive",
    "gamma": 0.99,
    "lam": 0.95,
    "desired_kl": 0.01,
    "max_grad_norm": 1.0,
}


def train_cfg(wandb: bool = False) -> dict:
    """The ``OnPolicyRunner`` configuration; ``wandb`` adds Weights & Biases logging.

    The device and the iteration count are arguments of the runner and of ``learn()``; the seed is
    the environment's (Isaac Lab seeds Python, numpy and torch with it).
    """
    return {
        "num_steps_per_env": NUM_STEPS_PER_ENV,
        "save_interval": SAVE_INTERVAL,
        "logger": (
            {"class_name": "WandbLogWriter", "project_name": WANDB_PROJECT}
            if wandb
            else "tensorboard"
        ),
        "obs_groups": {"actor": ["policy"], "critic": ["critic"]},
        "actor": copy.deepcopy(ACTOR),
        "critic": copy.deepcopy(CRITIC),
        "algorithm": copy.deepcopy(ALGORITHM),
    }


def load_actor(path: str, obs_dim: int, act_dim: int, device: str):
    """The actor of a training checkpoint as an rsl_rl model in eval mode.

    Rebuilds rsl_rl's own ``MLPModel``, so eval runs the bit-identical training actor; the robot
    uses the torch-only :mod:`dexport.skills.rotation.actor`, which matches it to ~1e-6. Call it as
    ``actor({"policy": obs}, stochastic_output=False)`` for the mean action.
    """
    import torch
    from rsl_rl.utils import resolve_callable

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    cfg = copy.deepcopy(ACTOR)
    model_class = resolve_callable(cfg.pop("class_name"))
    distribution_cfg = cfg.pop("distribution_cfg")
    obs = {"policy": torch.zeros(1, obs_dim, device=device)}
    actor = model_class(
        obs, {"actor": ["policy"]}, "actor", act_dim, distribution_cfg=distribution_cfg, **cfg
    ).to(device)
    actor.load_state_dict(checkpoint["actor_state_dict"])
    actor.eval()
    return actor
