# src/deep_svdd.py

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------
# Feed-forward network used by Deep SVDD
# ----------------------------------------------------
class DeepSVDDNet(nn.Module):
    def __init__(self, input_dim, hidden_dims=(4096, 1024), rep_dim=128):
        super().__init__()
        dims = [input_dim] + list(hidden_dims) + [rep_dim]

        layers = []
        for i in range(len(dims) - 1):
            in_d, out_d = dims[i], dims[i + 1]
            # CRITICAL: no bias terms in Deep SVDD network
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
# Symmetric Autoencoder (for pre-training)
# ----------------------------------------------------
class AutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dims=(4096, 1024), rep_dim=128):
        super().__init__()
        dims = [input_dim] + list(hidden_dims) + [rep_dim]

        enc_layers = []
        for i in range(len(dims) - 1):
            enc_layers.append(nn.Linear(dims[i], dims[i + 1], bias=True))
            if i < len(dims) - 2:
                enc_layers.append(nn.ReLU())
        self.encoder = nn.Sequential(*enc_layers)

        dec_dims = list(reversed(dims))
        dec_layers = []
        for i in range(len(dec_dims) - 1):
            dec_layers.append(nn.Linear(dec_dims[i], dec_dims[i + 1], bias=True))
            if i < len(dec_dims) - 2:
                dec_layers.append(nn.ReLU())
        self.decoder = nn.Sequential(*dec_layers)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
        z = self.encoder(x)
        x_rec = self.decoder(z)
        return x_rec

    def encode(self, x):
        return self.encoder(x)


# ----------------------------------------------------
# Deep SVDD wrapper (as in Ruff et al.)
# ----------------------------------------------------
class DeepSVDD:
    def __init__(
        self,
        objective,
        nu,
        input_dim,
        hidden_dims,
        rep_dim,
        device="cuda",
    ):
        assert objective in ("one-class", "soft-boundary")
        self.objective = objective
        self.nu = nu

        self.device = torch.device(device)
        self.net = DeepSVDDNet(input_dim, hidden_dims, rep_dim).to(self.device)

        self.c = None  # center
        self.R = torch.tensor(0.0, device=self.device)  # radius

    # ------------------------------
    # Weight init from Autoencoder
    # ------------------------------
    def load_weights_from_ae(self, ae: AutoEncoder):
        """Copy encoder weights layer-by-layer from AE into DeepSVDDNet."""
        ae_state = ae.encoder.state_dict()
        svdd_state = self.net.state_dict()

        # Map by order of linear layers
        new_state = {}
        ae_keys = [k for k in ae_state.keys() if "weight" in k]
        svdd_keys = [k for k in svdd_state.keys() if "weight" in k]

        for ae_k, svdd_k in zip(ae_keys, svdd_keys):
            new_state[svdd_k] = ae_state[ae_k]

        self.net.load_state_dict(new_state, strict=False)

    # ------------------------------
    # Center initialization
    # ------------------------------
    @torch.no_grad()
    def init_center_c(self, train_loader, eps=0.1):
        self.net.eval()
        n_samples = 0
        c = torch.zeros(self.net.net[-1].out_features, device=self.device)

        for (x_batch,) in train_loader:
            x_batch = x_batch.to(self.device)
            outputs = self.net(x_batch)
            n_samples += outputs.shape[0]
            c += outputs.sum(dim=0)

        c /= n_samples

        # If any dimension is too close to zero, push it slightly
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c >= 0)] = eps

        self.c = c.detach()

    # ------------------------------
    # Save model + scaler
    # ------------------------------
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
