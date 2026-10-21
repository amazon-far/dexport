"""Observation encoders: a fine-tuned DINOv2 backbone for images, an MLP per proprio key."""

import torch
from torch import nn


class MLP(nn.Module):
    """Linear + ReLU per layer (including the last)."""

    def __init__(self, units, input_size):
        super().__init__()
        layers = []
        for output_size in units:
            layers.append(nn.Linear(input_size, output_size))
            layers.append(nn.ReLU())
            input_size = output_size
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class StateEncoder(nn.Module):
    """Encoder of one proprio key; the nesting gives the checkpoint keys
    ``<key>_encoder.mlp.mlp.<i>``."""

    def __init__(self, input_dim, net_dim):
        super().__init__()
        self.mlp = MLP(net_dim, input_dim)

    def forward(self, x):
        return self.mlp(x)


def load_dinov2(name: str, pretrained: bool = True) -> nn.Module:
    """A DINOv2 ViT from torch.hub; its forward returns the CLS embedding (``embed_dim`` wide).

    ``pretrained=False`` builds the architecture only (no weight download), for loading a
    trained checkpoint; the hub code itself must be cached or downloadable.
    """
    return torch.hub.load("facebookresearch/dinov2", name, pretrained=pretrained)
