"""Torch-only loader for the rotation-skill actor saved by ``sim/train.py`` (RSL-RL format).

The checkpoint's ``actor_state_dict`` holds an MLP (``mlp.{0,2,4,...}.{weight,bias}``, ELU between
layers) and, when trained with observation normalization, ``obs_normalizer._mean/_std``. The actor
is rebuilt from those tensors, so neither rsl_rl nor Isaac Lab is needed on the robot: normalize
``(x - mean) / (std + 1e-2)``, run the MLP, return its output (the mean of the action
distribution). Layer widths are read from the tensor shapes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError as exc:
    raise ImportError(
        "the in-hand rotation skill needs torch (CPU is enough): pip install torch"
    ) from exc


class RotationActor(nn.Module):
    """Deterministic actor: optional observation normalization, then the MLP."""

    def __init__(self, state_dict: dict):
        super().__init__()
        lin_idxs = sorted(
            int(k.split(".")[1])
            for k in state_dict
            if k.startswith("mlp.") and k.endswith(".weight")
        )
        if not lin_idxs:
            raise ValueError("actor_state_dict has no mlp.*.weight entries (not an MLP actor)")
        layers: list[nn.Module] = []
        for n, i in enumerate(lin_idxs):
            out_f, in_f = state_dict[f"mlp.{i}.weight"].shape
            layers.append(nn.Linear(in_f, out_f))
            if n < len(lin_idxs) - 1:
                layers.append(nn.ELU())
        self.mlp = nn.Sequential(*layers)
        self.obs_dim = int(state_dict[f"mlp.{lin_idxs[0]}.weight"].shape[1])
        self.act_dim = int(state_dict[f"mlp.{lin_idxs[-1]}.weight"].shape[0])

        self.obs_normalization = "obs_normalizer._mean" in state_dict
        if self.obs_normalization:
            self.register_buffer(
                "_norm_mean", state_dict["obs_normalizer._mean"].reshape(1, -1).float()
            )
            self.register_buffer(
                "_norm_std", state_dict["obs_normalizer._std"].reshape(1, -1).float()
            )
            self._norm_eps = 1e-2  # rsl_rl EmpiricalNormalization eps

        self.mlp.load_state_dict(
            {k[len("mlp.") :]: v for k, v in state_dict.items() if k.startswith("mlp.")}
        )
        self.eval()

    @torch.no_grad()
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if self.obs_normalization:
            obs = (obs - self._norm_mean) / (self._norm_std + self._norm_eps)
        return self.mlp(obs)

    @torch.no_grad()
    def act_np(self, obs_1d: np.ndarray) -> np.ndarray:
        """``(obs_dim,)`` array -> ``(act_dim,)`` action."""
        x = torch.as_tensor(np.asarray(obs_1d, dtype=np.float32)).reshape(1, -1)
        return self.forward(x).squeeze(0).cpu().numpy()

    def obs_norm_mean(self) -> np.ndarray | None:
        """The normalizer mean, shape ``(obs_dim,)``, or None without normalization."""
        if not self.obs_normalization:
            return None
        return self._norm_mean.reshape(-1).cpu().numpy()


def load_rotation_actor(ckpt_path: str | Path, device: str = "cpu") -> RotationActor:
    """Load the actor of an RSL-RL checkpoint (``model_*.pt``)."""
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    if "actor_state_dict" not in ckpt:
        raise ValueError(
            f"{ckpt_path}: no 'actor_state_dict' (not an RSL-RL checkpoint); "
            f"top-level keys: {list(ckpt.keys())}"
        )
    return RotationActor(ckpt["actor_state_dict"]).to(device)
