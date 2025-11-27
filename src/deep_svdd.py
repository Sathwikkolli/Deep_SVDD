# src/deep_svdd.py

import torch
import torch.nn as nn


# ----------------------------------------------------
# Feed-forward encoder for Deep-SVDD (NO BIAS)
# ----------------------------------------------------
class DeepSVDDNet(nn.Module):
    def __init__(self, input_dim, hidden_dims=(2048, 512), rep_dim=128):
        super().__init__()

        dims = [input_dim] + list(hidden_dims) + [rep_dim]

        layers = []
        for i in range(len(dims) - 1):
            in_d, out_d = dims[i], dims[i + 1]

            layers.append(nn.Linear(in_d, out_d, bias=False))

            if i < len(dims) - 2:
                layers.append(nn.LeakyReLU(0.1))

        self.net = nn.Sequential(*layers)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
        return self.net(x)


# ----------------------------------------------------
# Deep-SVDD Wrapper (Ruff et al. 2018)
# ----------------------------------------------------
class DeepSVDD:
    def __init__(
        self,
        objective,
        nu,
        input_dim,
        hidden_dims=(2048, 512),
        rep_dim=128,
        device="cuda",
    ):
        assert objective in ("one-class", "soft-boundary")
        self.objective = objective
        self.nu = nu

        self.device = torch.device(device)

        self.net = DeepSVDDNet(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            rep_dim=rep_dim,
        ).to(self.device)

        self.c = None
        self.R = torch.tensor(0.0, device=self.device)

    # ----------------------------------------------------
    # Initialize center c
    # ----------------------------------------------------
    @torch.no_grad()
    def init_center_c(self, train_loader, eps=0.1):
        self.net.eval()

        rep_dim = self.net.net[-1].out_features
        c = torch.zeros(rep_dim, device=self.device)
        n_samples = 0

        for (x_batch,) in train_loader:
            x_batch = x_batch.to(self.device)
            outputs = self.net(x_batch)
            c += outputs.sum(dim=0)
            n_samples += outputs.size(0)

        c /= n_samples

        # Avoid near-zero components
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c >= 0)] = eps

        self.c = c.detach()

    # ----------------------------------------------------
    # Save
    # ----------------------------------------------------
    def save_state(self, path, scaler=None):
        state = {
            "net_state_dict": self.net.state_dict(),
            "center_c": self.c,
            "radius_R": self.R,
            "objective": self.objective,
            "nu": self.nu,
            "scaler": scaler,
        }
        torch.save(state, path)
