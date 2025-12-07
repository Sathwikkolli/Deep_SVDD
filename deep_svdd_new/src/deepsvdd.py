# File: deepsvdd.py

import torch
import torch.nn as nn


class DeepSVDDNet(nn.Module):
    """
    MLP encoder for Deep-SVDD.
    - input_dim: 1024 (from WavLM)
    - hidden: list or tuple, e.g. [] or [768]
    - rep_dim: 512 (latent representation)
    No bias, LeakyReLU activations (except last layer).
    """
    def __init__(self, input_dim=1024, hidden=None, rep_dim=512):
        super().__init__()
        if hidden is None:
            hidden = []

        dims = [input_dim] + list(hidden) + [rep_dim]
        layers = []

        for i in range(len(dims) - 1):
            in_d, out_d = dims[i], dims[i + 1]
            layers.append(nn.Linear(in_d, out_d, bias=False))
            if i < len(dims) - 2:
                layers.append(nn.LeakyReLU(0.1))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
