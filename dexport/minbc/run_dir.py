"""Run directory layout: ``config.json``, ``stats.pkl``, ``norm.pkl``, ``model_{last,best}.ckpt``.

A checkpoint is the plain ``state_dict`` of the EMA network. ``stats.pkl`` (action and
relative-EE bounds) and ``norm.pkl`` (observation percentiles) are the normalization the run was
trained with; inference reads all of them from the checkpoint's directory.
"""

import dataclasses
import json
import os
import pickle

from dexport.minbc.config import MinBCConfig, config_from_dict

CONFIG_FILE = "config.json"
STATS_FILE = "stats.pkl"
NORM_FILE = "norm.pkl"
LAST_CKPT = "model_last.ckpt"
BEST_CKPT = "model_best.ckpt"


def save_config(run_dir: str, config: MinBCConfig, git_commit: str | None = None) -> None:
    """Write ``config.json`` (every field, plus the code version when known)."""
    d = dataclasses.asdict(config)
    if git_commit:
        d["git_commit"] = git_commit
    with open(os.path.join(run_dir, CONFIG_FILE), "w") as f:
        json.dump(d, f, indent=4)


def load_config(run_dir: str) -> MinBCConfig:
    with open(os.path.join(run_dir, CONFIG_FILE)) as f:
        return config_from_dict(json.load(f))


def save_stats(run_dir: str, stats: dict, obs_norm: dict) -> None:
    with open(os.path.join(run_dir, STATS_FILE), "wb") as f:
        pickle.dump(stats, f)
    with open(os.path.join(run_dir, NORM_FILE), "wb") as f:
        pickle.dump(obs_norm, f)


def load_stats(run_dir: str) -> tuple[dict, dict]:
    """``(stats, obs_norm)`` from ``stats.pkl`` and ``norm.pkl``."""
    return (
        load_pickle(os.path.join(run_dir, STATS_FILE)),
        load_pickle(os.path.join(run_dir, NORM_FILE)),
    )


class _NumpyCompatUnpickler(pickle.Unpickler):
    """Resolve the ``numpy._core`` module path of numpy>=2 pickles on numpy<2."""

    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core", 1)
        return super().find_class(module, name)


def load_pickle(path: str):
    with open(path, "rb") as f:
        try:
            return pickle.load(f)
        except ModuleNotFoundError:
            f.seek(0)
            return _NumpyCompatUnpickler(f).load()


def resolve_checkpoint(path: str) -> tuple[str, str]:
    """``(run_dir, ckpt_path)`` for a run directory (its ``model_last.ckpt``) or a checkpoint."""
    path = os.path.abspath(os.path.expanduser(path))
    if os.path.isdir(path):
        ckpt = os.path.join(path, LAST_CKPT)
        if not os.path.isfile(ckpt):
            found = sorted(f for f in os.listdir(path) if f.endswith(".ckpt"))
            hint = f"; pass one of {found} explicitly" if found else ""
            raise FileNotFoundError(f"{path} has no {LAST_CKPT}{hint}")
        return path, ckpt
    if not os.path.isfile(path):
        raise FileNotFoundError(f"checkpoint not found: {path}")
    return os.path.dirname(path), path
