"""The trainable diffusion policy: network, EMA copy, noise scheduler, optimizer, one train step.

Training predicts the noise added to normalized action chunks (epsilon-prediction MSE). AdamW
trains the DINOv2 backbone at 0.1x the learning rate of the rest; the learning rate follows a
cosine schedule after ``WARMUP_STEPS`` linear warmup steps, stepped every batch. Checkpoints hold
the EMA network's weights only.
"""

import copy
import os

import torch
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from torch.nn.functional import mse_loss

from dexport.minbc.diffusion import make_noise_scheduler
from dexport.minbc.models.unet import ConditionalUnet1D

WARMUP_STEPS = 500


def make_ema(parameters) -> EMAModel:
    """EMA with decay ``min(1 - (1 + t) ** -0.75, 0.9999)`` at the update after ``t`` earlier ones
    (diffusers honours ``power`` only with ``use_ema_warmup``)."""
    return EMAModel(
        parameters=parameters, decay=0.9999, use_ema_warmup=True, inv_gamma=1.0, power=0.75
    )


class DiffusionPolicy:
    """Training-time policy (network, EMA copy, optimizer); inference loads checkpoints through
    ``dexport.minbc.inference.DiffusionAgent``."""

    def __init__(self, config, device="cpu"):
        self.config = config
        self.device = device
        self.model = ConditionalUnet1D(config)
        self.noise_scheduler = make_noise_scheduler(
            config.dp.diffusion_method, config.dp.diffusion_iters
        )
        self.ema = make_ema(self.model.parameters())
        # A separate network that receives the EMA weights (for evaluation and checkpoints).
        self.ema_model = copy.deepcopy(self.model)
        self.optimizer = self._make_optimizer(config.optim.learning_rate, config.optim.weight_decay)
        self.lr_scheduler = None
        self.to(device)

    def _make_optimizer(self, lr, weight_decay):
        if "img_encoder" not in self.model.encoders:
            return torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        image_params = list(self.model.encoders["img_encoder"].parameters())
        image_ids = {id(p) for p in image_params}
        other_params = [p for p in self.model.parameters() if id(p) not in image_ids]
        return torch.optim.AdamW(
            [{"params": other_params, "lr": lr}, {"params": image_params, "lr": lr / 10}],
            weight_decay=weight_decay,
        )

    def set_lr_scheduler(self, num_training_steps: int):
        self.lr_scheduler = get_scheduler(
            name="cosine",
            optimizer=self.optimizer,
            num_warmup_steps=WARMUP_STEPS,
            num_training_steps=num_training_steps,
        )

    def to(self, device):
        self.device = device
        self.model.to(device)
        self.ema.to(device)
        self.ema_model.to(device)

    def loss(self, batch) -> torch.Tensor:
        """Noise-prediction MSE at a random diffusion step per sample."""
        actions = batch["action"].to(self.device)
        noise = torch.randn(actions.shape, device=self.device)
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (actions.shape[0],),
            device=self.device,
        ).long()
        noisy_actions = self.noise_scheduler.add_noise(actions, noise, timesteps)
        return mse_loss(self.model(batch, noisy_actions, timesteps), noise)

    def train_step(self, batch) -> dict:
        """One gradient step (optimizer, LR schedule, EMA); returns the logged scalars."""
        if self.lr_scheduler is None:
            raise RuntimeError("call set_lr_scheduler(num_training_steps) before train_step")
        loss = self.loss(batch)
        loss.backward()
        lr = self.optimizer.param_groups[0]["lr"]
        params = list(self.model.parameters())
        grad_norm = torch.norm(
            torch.stack([p.grad.detach().norm(2) for p in params if p.grad is not None]), 2
        ).item()
        weight_norm = torch.norm(torch.stack([p.detach().norm(2) for p in params]), 2).item()
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.lr_scheduler.step()
        self.ema.step(self.model.parameters())
        return {"loss": loss.item(), "lr": lr, "grad_norm": grad_norm, "weight_norm": weight_norm}

    def copy_ema_weights(self):
        """Copy the EMA weights into ``ema_model``; the training weights are untouched."""
        self.ema.copy_to(self.ema_model.parameters())

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.copy_ema_weights()
        torch.save(self.ema_model.state_dict(), path)
