"""Load a trained run and predict action chunks (the deploy-facing API).

``DiffusionAgent.load(path)`` takes a run directory (its ``model_last.ckpt``) or a checkpoint
file, rebuilds the network from the run's ``config.json`` and loads the EMA weights and the
``stats.pkl`` / ``norm.pkl`` normalization, independent of the working directory.
``predict_chunk(window)`` maps one observation window (``obs_horizon`` per-step dicts as built by
:func:`dexport.deploy.obs.obs_window`: proprio arrays, plus every camera of the step under
``WINDOW_IMAGES_KEY`` when the policy uses images) to the unnormalized (total_horizon, action_dim)
chunk. ``set_sampler`` swaps the reverse-diffusion sampler (the deploy default is DDIM with 10
steps).
"""

import numpy as np
import torch

from dexport.minbc.config import WINDOW_IMAGES_KEY
from dexport.minbc.diffusion import make_noise_scheduler, sample_actions, scheduler_method
from dexport.minbc.models.unet import ConditionalUnet1D
from dexport.minbc.normalization import (
    action_bounds,
    minmax_unnorm_data,
    normalize_obs,
    obs_clip_limit,
)
from dexport.minbc.run_dir import load_config, load_stats, resolve_checkpoint
from dexport.minbc.transforms import build_image_transforms


def _image_tensor(step) -> torch.Tensor:
    """float32 (n_cam, 3, H, W) tensor of one window step's (n_cam, H, W, 3) camera frames."""
    frames = np.asarray(step[WINDOW_IMAGES_KEY], dtype=np.float32)
    return torch.from_numpy(frames).permute(0, 3, 1, 2)


class DiffusionAgent:
    """Inference-time policy: a trained network and its normalization on one device.

    Not thread-safe: inference mutates the noise scheduler.
    """

    def __init__(self, config, model, stats, obs_norm, device="cpu"):
        self.config = config
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.stats = stats
        self.obs_norm = obs_norm
        _, self.eval_transform = build_image_transforms(config)
        self.noise_scheduler = make_noise_scheduler(
            config.dp.diffusion_method, config.dp.diffusion_iters
        )
        self.num_inference_steps = int(config.dp.diffusion_iters)
        self._clip = obs_clip_limit(config.data.norm_percentile)

    @classmethod
    def load(cls, path: str, device="cuda:0") -> "DiffusionAgent":
        run_dir, ckpt = resolve_checkpoint(path)
        config = load_config(run_dir)
        stats, obs_norm = load_stats(run_dir)
        # The checkpoint holds every weight, so the backbone is built without pretrained weights.
        model = ConditionalUnet1D(config, pretrained_backbone=False)
        model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
        return cls(config, model, stats, obs_norm, device)

    @property
    def obs_horizon(self) -> int:
        return self.config.dp.obs_horizon

    @property
    def total_horizon(self) -> int:
        return self.config.dp.total_horizon

    @property
    def action_dim(self) -> int:
        return self.config.data.action_dim

    def set_sampler(self, method: str | None = None, steps: int | None = None) -> None:
        """Use ``method`` ("ddim" / "ddpm"; None = as trained) with ``steps`` reverse steps
        (None = the training steps). The weights predict epsilon on the training schedule, so the
        new scheduler keeps the training timesteps and only subsamples them."""
        trained_steps = int(self.config.dp.diffusion_iters)
        method = method or self.config.dp.diffusion_method
        if method != scheduler_method(self.noise_scheduler):
            self.noise_scheduler = make_noise_scheduler(method, trained_steps)
        n = trained_steps if steps is None else int(steps)
        if n < 1:
            raise ValueError(f"the number of diffusion steps must be >= 1; got {n}")
        if n > trained_steps:
            print(
                f"[minbc] {n} diffusion steps exceed the {trained_steps} the policy was trained "
                f"with; using {trained_steps}.",
                flush=True,
            )
            n = trained_steps
        self.num_inference_steps = n

    def prepare_obs(self, window) -> dict:
        """Normalized network input ``{key: (1, obs_horizon, ...)}`` on the agent's device."""
        if len(window) != self.obs_horizon:
            raise ValueError(f"expected {self.obs_horizon} observation steps, got {len(window)}")
        obs = {}
        for key in self.config.data.data_key:
            if key == "img":
                frames = [self.eval_transform(_image_tensor(step)) for step in window]
                obs["img"] = torch.stack(frames).to(self.device)[None]
            else:
                x = np.stack([step[key] for step in window])
                x = normalize_obs(key, x, self.stats, self.obs_norm, self._clip)
                obs[key] = torch.from_numpy(np.ascontiguousarray(x)).to(
                    self.device, dtype=torch.float32
                )[None]
        return obs

    def unnormalize_actions(self, x_norm: np.ndarray) -> np.ndarray:
        """Normalized (..., action_dim) actions -> robot units."""
        return minmax_unnorm_data(x_norm, *action_bounds(self.stats, self.action_dim))

    @torch.no_grad()
    def predict_chunk(self, window) -> np.ndarray:
        """The unnormalized (total_horizon, action_dim) chunk for one observation window."""
        obs_cond = self.model.forward_encoder(self.prepare_obs(window))
        x = sample_actions(
            self.model,
            obs_cond,
            self.noise_scheduler,
            self.num_inference_steps,
            (1, self.total_horizon, self.action_dim),
        )
        return self.unnormalize_actions(x[0].cpu().numpy())
