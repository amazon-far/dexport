"""Noise schedulers and the reverse-diffusion sampling loop (training eval and inference)."""

import torch
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

SCHEDULERS = {"ddim": DDIMScheduler, "ddpm": DDPMScheduler}


def make_noise_scheduler(method: str, num_train_timesteps: int):
    """The training scheduler: squared-cosine betas, clipped samples, epsilon prediction."""
    return SCHEDULERS[method](
        num_train_timesteps=int(num_train_timesteps),
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        prediction_type="epsilon",
    )


def scheduler_method(scheduler) -> str:
    """``"ddim"`` or ``"ddpm"`` for an installed scheduler."""
    return "ddim" if isinstance(scheduler, DDIMScheduler) else "ddpm"


@torch.no_grad()
def sample_actions(model, obs_cond, scheduler, num_steps: int, shape) -> torch.Tensor:
    """Denoise one ``torch.randn(shape)`` draw in ``num_steps`` reverse steps (normalized actions).

    Mutates ``scheduler`` (its timesteps), so concurrent calls must not share one.
    """
    device = obs_cond.device
    x = torch.randn(shape, device=device)
    scheduler.set_timesteps(num_steps, device=device)
    scheduler.alphas_cumprod = scheduler.alphas_cumprod.to(device)
    for k in scheduler.timesteps:
        if k.ndim == 0:
            k = k[None].to(device)
        eps = model.forward_denoise(obs_cond, sample=x, timestep=k)
        x = scheduler.step(model_output=eps, timestep=k, sample=x).prev_sample
    return x
