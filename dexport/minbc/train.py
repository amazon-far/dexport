"""Diffusion-policy training CLI.

    python -m dexport.minbc.train <preset> --train-data TRAIN --test-data TEST [overrides]

Presets are in ``presets.py``; any field can be overridden (``--data.hand-action-source state``,
``--optim.num-train-steps 1000``, ...). The run directory ``<output_dir>/<output_name>`` receives
``config.json``, ``stats.pkl``, ``norm.pkl``, tensorboard logs, ``model_last.ckpt`` (every
``save_freq`` steps) and ``model_best.ckpt`` (the lowest test action MSE; the test split is
evaluated every ``eval_freq`` steps, the train split every ``TRAIN_EVAL_EVERY * eval_freq``).
A console summary is printed every ``min(eval_freq, save_freq)`` steps. A run directory that
already holds a checkpoint is refused. ``--use-wandb`` also logs to Weights & Biases
(``WANDB_MODE=offline`` works).
"""

import dataclasses
import math
import os
import random
import subprocess
import time

import numpy as np
import torch
import torch.utils.data
import tyro
from torch.nn.functional import mse_loss

from dexport.minbc.config import MinBCConfig
from dexport.minbc.data.dataset import WindowDataset
from dexport.minbc.data.sampling import make_train_sampler
from dexport.minbc.data.zarr_io import list_trajectories
from dexport.minbc.diffusion import sample_actions
from dexport.minbc.normalization import action_bounds, minmax_norm_data, minmax_unnorm_data
from dexport.minbc.policy import DiffusionPolicy
from dexport.minbc.presets import PRESETS
from dexport.minbc.run_dir import BEST_CKPT, LAST_CKPT, save_config, save_stats
from dexport.minbc.transforms import build_image_transforms

#: The train split is evaluated at every this-many-th test evaluation (and at the last step).
TRAIN_EVAL_EVERY = 5


def set_seed(seed: int) -> None:
    """Seed python, numpy and torch and make cuDNN deterministic."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)  # for subprocesses: this interpreter's is fixed
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def git_commit() -> str | None:
    """Short commit hash of the checkout this package runs from, or None outside a git checkout."""
    try:
        out = subprocess.run(
            [
                "git",
                "-C",
                os.path.dirname(os.path.abspath(__file__)),
                "rev-parse",
                "--short",
                "HEAD",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip() or None


@dataclasses.dataclass
class TrainingData:
    train_loader: torch.utils.data.DataLoader
    test_loader: torch.utils.data.DataLoader
    stats: dict
    obs_norm: dict
    train_traj_paths: list
    test_traj_paths: list


def build_training_data(config: MinBCConfig, pin_memory: bool = False) -> TrainingData:
    """Train/test datasets and loaders. ``train_data`` may join several stores with '+'."""
    sources = [s.strip() for s in config.train_data.split("+")]
    if not all(sources):
        raise ValueError(f"train_data {config.train_data!r} has an empty '+'-joined part")
    if "+" in config.test_data:
        raise ValueError(f"test_data {config.test_data!r} must be a single store")
    train_paths, train_sources = [], []
    for i, source in enumerate(sources):
        trajs = list_trajectories(os.path.join(config.data_dir, source))
        train_paths += trajs
        train_sources += [i] * len(trajs)
    test_paths = list_trajectories(os.path.join(config.data_dir, config.test_data.strip()))

    train_tf, eval_tf = build_image_transforms(config)
    train_set = WindowDataset(config, train_paths, "train", train_sources, transform=train_tf)
    # A source whose trajectories were all skipped (or are too short) would silently drop out
    # of the co-training mix.
    windows = np.bincount(train_set.source_ids, minlength=len(sources))
    empty = [source for source, n in zip(sources, windows) if n == 0]
    if empty:
        raise ValueError(
            f"train_data source(s) {empty} contribute no training windows (unreadable or too "
            f"short trajectories, see the warnings above)"
        )
    test_set = WindowDataset(
        config,
        test_paths,
        "test",
        stats=train_set.stats,
        obs_norm=train_set.obs_norm,
        transform=eval_tf,
    )
    if len(test_set) == 0:
        raise ValueError(
            f"test_data {config.test_data!r} has no windows (every episode is shorter than "
            f"{test_set.min_episode_frames} frames after the tracking-jump splits); "
            "model_best.ckpt is selected on it"
        )
    sampler = make_train_sampler(config, train_set)
    workers = config.optim.num_workers
    loader_args = dict(
        batch_size=config.optim.batch_size,
        num_workers=workers,
        pin_memory=pin_memory,
        persistent_workers=workers > 0,
    )
    train_loader = torch.utils.data.DataLoader(
        train_set, sampler=sampler, shuffle=sampler is None, drop_last=True, **loader_args
    )
    if len(train_loader) == 0:
        raise ValueError(
            f"{len(train_set)} training windows do not fill one batch of {config.optim.batch_size}"
        )
    test_loader = torch.utils.data.DataLoader(test_set, shuffle=False, **loader_args)
    return TrainingData(
        train_loader, test_loader, train_set.stats, train_set.obs_norm, train_paths, test_paths
    )


def action_errors(pred_norm, gt_norm, act_min, act_max) -> dict:
    """Batch sums of per-sample MSEs: ``action_mse`` and the arm's ``ee_mse`` (dims 0-2) and
    ``rot6d_mse`` (dims 3-8) in action units, ``normalized_mse`` in the [-1, 1] space."""
    pred = minmax_unnorm_data(pred_norm, act_min, act_max)
    gt = minmax_unnorm_data(gt_norm, act_min, act_max)
    per_elem = mse_loss(torch.tensor(pred), torch.tensor(gt), reduction="none")
    normalized = mse_loss(
        torch.tensor(minmax_norm_data(pred, act_min, act_max)),
        torch.tensor(minmax_norm_data(gt, act_min, act_max)),
        reduction="none",
    )
    return {
        "action_mse": per_elem.mean(-1).mean(-1).sum().item(),
        "normalized_mse": normalized.mean(-1).mean(-1).sum().item(),
        "ee_mse": per_elem[..., :3].mean(-1).mean(-1).sum().item(),
        "rot6d_mse": per_elem[..., 3:9].mean(-1).mean(-1).sum().item(),
    }


@torch.no_grad()
def evaluate(policy: DiffusionPolicy, loader, stats: dict, num_steps: int) -> dict:
    """Mean ``action_errors`` of chunks sampled with the current EMA weights over a loader."""
    policy.copy_ema_weights()
    model = policy.ema_model.eval()
    d = policy.config.data.action_dim
    act_min, act_max = action_bounds(stats, d)
    totals = dict.fromkeys(("action_mse", "normalized_mse", "ee_mse", "rot6d_mse"), 0.0)
    count = 0
    for batch in loader:
        gt = batch["action"]
        obs_cond = model.forward_encoder(batch)
        shape = (gt.shape[0], policy.config.dp.total_horizon, d)
        pred = sample_actions(model, obs_cond, policy.noise_scheduler, num_steps, shape)
        for key, value in action_errors(pred.cpu().numpy(), gt.numpy(), act_min, act_max).items():
            totals[key] += value
        count += gt.shape[0]
    return {key: value / count for key, value in totals.items()}


def _start_wandb(config: MinBCConfig, run_dir: str):
    try:
        import wandb
    except ImportError as e:
        raise ImportError("--use-wandb needs the wandb package (pip install wandb)") from e
    return wandb.init(
        project=config.wandb_project,
        entity=config.wandb_entity,
        name=config.output_name,
        dir=run_dir,
        config=dataclasses.asdict(config),
    )


def _forever(loader):
    while True:
        yield from loader


def _new_log_interval() -> dict:
    """Accumulators of the steps since the last console summary."""
    return {"losses": [], "data_s": 0.0, "compute_s": 0.0, "t0": time.time()}


def _log_interval(interval: dict, step: int, total: int, steps_per_epoch: int, t_start: float, log):
    """Console line and scalars summarizing ``interval`` (the steps since the last summary)."""
    interval_s = time.time() - interval["t0"]
    eta_h = (time.time() - t_start) / step * (total - step) / 3600
    data_s, compute_s = interval["data_s"], interval["compute_s"]
    data_frac = data_s / max(interval_s, 1e-9)
    loss = float(np.mean(interval["losses"]))
    print(
        f"step {step}/{total} | epoch {step / steps_per_epoch:.2f} | loss {loss:.4f} | "
        f"data {data_s:.1f}s ({100 * data_frac:.0f}%) / compute {compute_s:.1f}s | "
        f"ETA {eta_h:.1f}h",
        flush=True,
    )
    log(
        {
            "train/interval_loss": loss,
            "time/interval_s": interval_s,
            "time/data_s_per_interval": data_s,
            "time/compute_s_per_interval": compute_s,
            "time/data_frac": data_frac,
        },
        step,
    )


def _evaluate_and_checkpoint(policy, data, step, best, num_steps, with_train, run_dir, log):
    """Evaluate the test split (a new best is saved as model_best.ckpt) and, ``with_train``,
    the train split. ``best`` holds the lowest action MSE so far per split."""
    name = policy.config.output_name
    t_eval = time.time()
    test = evaluate(policy, data.test_loader, data.stats, num_steps)
    if test["action_mse"] < best["test"]:
        best["test"] = test["action_mse"]
        policy.save(os.path.join(run_dir, BEST_CKPT))
    log(
        {
            **{f"eval/{k}": v for k, v in test.items()},
            "eval/best_action_mse": best["test"],
            "time/eval_s": time.time() - t_eval,
        },
        step,
    )
    print(
        f"{name} | step {step} | test action MSE {test['action_mse']:.4f} "
        f"(best {best['test']:.4f}) | EE {test['ee_mse']:.4f} | rot6d {test['rot6d_mse']:.4f}",
        flush=True,
    )
    if not with_train:
        return
    train = evaluate(policy, data.train_loader, data.stats, num_steps)
    best["train"] = min(best["train"], train["action_mse"])
    log(
        {
            **{f"eval_train/{k}": v for k, v in train.items()},
            "eval_train/best_action_mse": best["train"],
        },
        step,
    )
    print(
        f"{name} | step {step} | train action MSE {train['action_mse']:.4f} "
        f"(best {best['train']:.4f})",
        flush=True,
    )


def _fit(policy: DiffusionPolicy, data: TrainingData, run_dir: str, log) -> None:
    dp, optim = policy.config.dp, policy.config.optim
    total = optim.num_train_steps
    eval_steps = dp.eval_diffusion_iters or dp.diffusion_iters
    log_every = min(optim.eval_freq, optim.save_freq)
    steps_per_epoch = max(len(data.train_loader), 1)
    batches = _forever(data.train_loader)
    best = {"test": math.inf, "train": math.inf}
    interval = _new_log_interval()
    t_start = t_data = time.time()
    policy.model.train()
    for step in range(total):
        batch = next(batches)
        t_compute = time.time()
        metrics = policy.train_step(batch)
        data_s, compute_s = t_compute - t_data, time.time() - t_compute
        interval["losses"].append(metrics["loss"])
        interval["data_s"] += data_s
        interval["compute_s"] += compute_s
        log(
            {
                "train/loss": metrics["loss"],
                "train/lr": metrics["lr"],
                "train/grad_norm": metrics["grad_norm"],
                "train/weight_norm": metrics["weight_norm"],
                "time/data_s_per_step": data_s,
                "time/compute_s_per_step": compute_s,
                "train/epoch": step / steps_per_epoch,
            },
            step,
        )
        done = step + 1
        last = done == total
        t_data = time.time()
        if done % log_every == 0 or last:
            _log_interval(interval, done, total, steps_per_epoch, t_start, log)
            interval = _new_log_interval()
        if done % optim.save_freq == 0 or last:
            policy.save(os.path.join(run_dir, LAST_CKPT))
        if done % optim.eval_freq == 0 or last:
            with_train = done % (TRAIN_EVAL_EVERY * optim.eval_freq) == 0 or last
            _evaluate_and_checkpoint(policy, data, done, best, eval_steps, with_train, run_dir, log)
            t_data = time.time()


def train(config: MinBCConfig) -> str:
    """Train one policy; returns its run directory."""
    if not config.output_name:
        raise ValueError("set --output-name (the run directory is <output_dir>/<output_name>)")
    device = torch.device("cpu" if config.gpu == "cpu" else f"cuda:{config.gpu}")
    set_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    run_dir = os.path.join(config.output_dir, config.output_name)
    trained = [f for f in (LAST_CKPT, BEST_CKPT) if os.path.isfile(os.path.join(run_dir, f))]
    if trained:
        raise FileExistsError(
            f"{run_dir} already holds a trained run ({', '.join(trained)}); "
            f"choose another --output-name"
        )

    # The policy is built before the data (its seeded initialization); the run directory is
    # written once both exist, so a launch that fails while loading leaves nothing behind.
    policy = DiffusionPolicy(config, device)
    data = build_training_data(config, pin_memory=device.type == "cuda")
    os.makedirs(run_dir, exist_ok=True)
    save_config(run_dir, config, git_commit())
    save_stats(run_dir, data.stats, data.obs_norm)
    policy.set_lr_scheduler(config.optim.num_train_steps)
    n_params = sum(p.numel() for p in policy.model.parameters())
    print(
        f"[train] {n_params / 1e6:.1f}M parameters, {len(data.train_traj_paths)} train / "
        f"{len(data.test_traj_paths)} test trajectories -> {run_dir}",
        flush=True,
    )

    from torch.utils.tensorboard import SummaryWriter

    wandb_run = _start_wandb(config, run_dir) if config.use_wandb else None
    with SummaryWriter(run_dir) as writer:

        def log(scalars, step):
            for key, value in scalars.items():
                writer.add_scalar(key, value, step)
            if wandb_run is not None:
                wandb_run.log(scalars, step=step)

        try:
            _fit(policy, data, run_dir, log)
        finally:
            if wandb_run is not None:
                wandb_run.finish()
    return run_dir


def main(argv=None):
    train(
        tyro.extras.overridable_config_cli(PRESETS, prog="python -m dexport.minbc.train", args=argv)
    )


if __name__ == "__main__":
    main()
