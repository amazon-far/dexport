"""Noise-prediction network: observation encoders + a 1-D temporal U-Net with FiLM conditioning.

The U-Net follows Diffusion Policy (Chi et al., RSS 2023). The conditioning vector is, per
observation frame, the DINOv2 CLS features of every camera (interleaved: feature f of camera c
sits at ``f * n_cameras + c``) followed by one feature per proprio key in ``OBS_KEY_ORDER``,
flattened over frames, then prefixed with the diffusion-step embedding. Module names fix the
checkpoint's state_dict keys, and the registration order fixes their order.
"""

import math

import torch
from torch import nn

from dexport.minbc.config import OBS_KEY_ORDER, POLICY_OBS_DIMS
from dexport.minbc.models.encoders import StateEncoder, load_dinov2

# Fixed by the checkpoint format: diffusion-step embedding width, conv kernel, GroupNorm groups.
STEP_EMBED_DIM = 256
KERNEL_SIZE = 5
N_GROUPS = 8


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device) * -emb)
        emb = x[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Conv1dBlock(nn.Module):
    """Conv1d -> GroupNorm -> Mish."""

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class ConditionalResidualBlock1D(nn.Module):
    """Two conv blocks; the condition predicts a per-channel FiLM scale and bias in between."""

    def __init__(self, in_channels, out_channels, cond_dim, kernel_size=3, n_groups=8):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
                Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
            ]
        )
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(), nn.Linear(cond_dim, out_channels * 2), nn.Unflatten(-1, (-1, 1))
        )
        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x, cond):
        """x: (B, in_channels, T), cond: (B, cond_dim) -> (B, out_channels, T)."""
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond).reshape(x.shape[0], 2, self.out_channels, 1)
        out = embed[:, 0, ...] * out + embed[:, 1, ...]
        out = self.blocks[1](out)
        return out + self.residual_conv(x)


class ConditionalUnet1D(nn.Module):
    """Predicts the noise in a (B, total_horizon, action_dim) action chunk given observations."""

    def __init__(self, config, pretrained_backbone: bool = True):
        super().__init__()
        data = config.data
        self.obs_horizon = config.dp.obs_horizon
        self.obs_keys = tuple(k for k in OBS_KEY_ORDER if k in data.data_key)
        self.n_cameras = len(data.im_key)
        self.encoders = nn.ModuleDict()
        obs_dim = 0
        if "img" in self.obs_keys:
            backbone = load_dinov2(data.dinov2_model, pretrained_backbone)
            # One backbone shared by every camera; its weights repeat per camera in the state_dict.
            self.encoders["img_encoder"] = nn.ModuleList([backbone] * self.n_cameras)
            obs_dim += backbone.embed_dim * self.n_cameras
        for key in self.obs_keys:
            if key != "img":
                net_dim = getattr(config.dp.encoder, f"{key}_net_dim")
                self.encoders[f"{key}_encoder"] = StateEncoder(POLICY_OBS_DIMS[key], net_dim)
                obs_dim += net_dim[-1]

        action_dim = config.data.action_dim
        down_dims = list(config.dp.down_dims)
        all_dims = [action_dim] + down_dims
        dsed = STEP_EMBED_DIM
        cond_dim = dsed + obs_dim * self.obs_horizon
        in_out = list(zip(all_dims[:-1], all_dims[1:]))

        def block(dim_in, dim_out):
            return ConditionalResidualBlock1D(
                dim_in, dim_out, cond_dim=cond_dim, kernel_size=KERNEL_SIZE, n_groups=N_GROUPS
            )

        # Construction order (step encoder, mid, down, up, final) fixes the seeded init.
        diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed), nn.Linear(dsed, dsed * 4), nn.Mish(), nn.Linear(dsed * 4, dsed)
        )
        self.mid_modules = nn.ModuleList([block(all_dims[-1], all_dims[-1]) for _ in range(2)])
        down_modules = nn.ModuleList(
            nn.ModuleList(
                [
                    block(dim_in, dim_out),
                    block(dim_out, dim_out),
                    Downsample1d(dim_out) if ind < len(in_out) - 1 else nn.Identity(),
                ]
            )
            for ind, (dim_in, dim_out) in enumerate(in_out)
        )
        up_modules = nn.ModuleList(
            nn.ModuleList([block(dim_out * 2, dim_in), block(dim_in, dim_in), Upsample1d(dim_in)])
            for dim_in, dim_out in reversed(in_out[1:])
        )
        final_conv = nn.Sequential(
            Conv1dBlock(down_dims[0], down_dims[0], kernel_size=KERNEL_SIZE),
            nn.Conv1d(down_dims[0], action_dim, 1),
        )
        # Registered after mid_modules (and up before down) to match the trained checkpoints'
        # state_dict key order.
        self.diffusion_step_encoder = diffusion_step_encoder
        self.up_modules = up_modules
        self.down_modules = down_modules
        self.final_conv = final_conv

    def forward_encoder(self, obs):
        """Observation dict (B, To, ...) per key -> conditioning vector (B, To * obs_dim)."""
        device = next(self.parameters()).device
        features = []
        for key in self.obs_keys:
            x = obs[key][:, : self.obs_horizon].to(device)
            if key == "img":
                cams = self.encoders["img_encoder"]
                feats = [cams[i](x[:, :, i].flatten(end_dim=1)) for i in range(x.shape[2])]
                features.append(torch.stack(feats, dim=2).reshape(*x.shape[:2], -1))
            else:
                feat = self.encoders[f"{key}_encoder"](x.flatten(end_dim=1))
                features.append(feat.reshape(*x.shape[:2], -1))
        return torch.cat(features, dim=-1).flatten(start_dim=1)

    def forward_denoise(self, obs_cond, sample, timestep):
        """Noise (B, T, action_dim) in ``sample`` (B, T, action_dim) at ``timestep`` (a (B,) or
        scalar tensor, or an int), conditioned on ``obs_cond`` from :meth:`forward_encoder`."""
        sample = sample.moveaxis(-1, -2)
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        timesteps = timesteps.expand(sample.shape[0])

        global_feature = torch.cat([self.diffusion_step_encoder(timesteps), obs_cond], axis=-1)

        x = sample
        h = []
        for resnet, resnet2, downsample in self.down_modules:
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)
        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)
        for resnet, resnet2, upsample in self.up_modules:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)
        return self.final_conv(x).moveaxis(-1, -2)

    def forward(self, obs, sample, timestep):
        return self.forward_denoise(self.forward_encoder(obs), sample, timestep)
