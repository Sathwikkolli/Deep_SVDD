# src/deep_svdd.py
import torch
import torch.nn as nn
import torch.optim as optim


# ----------------------------------------------------
# Network φ(x)
# ----------------------------------------------------
class DeepSVDDNet(nn.Module):
    def __init__(self, input_dim=1024, rep_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, rep_dim)
        )

    def forward(self, x):
        return self.net(x)


# ----------------------------------------------------
# Deep SVDD Model
# ----------------------------------------------------
class DeepSVDD:
    def __init__(self, input_dim, rep_dim, objective="one-class", nu=0.1, device="cpu"):
        self.device = device
        self.objective = objective   # "one-class" or "soft-boundary"
        self.nu = nu

        self.net = DeepSVDDNet(input_dim=input_dim, rep_dim=rep_dim).to(device)

        # Hypersphere parameters
        self.c = None
        self.R = torch.tensor(0.0, device=device)   # Only used in soft-boundary

    # ------------------------------------------------
    # Initialize hypersphere center c
    # ------------------------------------------------
    @torch.no_grad()
    def init_center_c(self, train_loader, eps=0.1):
        n_samples = 0
        c = torch.zeros(self.net.net[-1].out_features, device=self.device)

        for (x,) in train_loader:
            x = x.to(self.device)
            z = self.net(x)
            n_samples += z.shape[0]
            c += z.sum(dim=0)

        c /= n_samples

        # No zero dims
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps

        self.c = c

    # ------------------------------------------------
    # Train procedure
    # ------------------------------------------------
    def train(self, train_loader, n_epochs, lr, weight_decay, print_every=1):
        optimizer = optim.Adam(self.net.parameters(), lr=lr, weight_decay=weight_decay)

        for epoch in range(1, n_epochs + 1):
            epoch_loss = 0.0
            n = 0
            self.net.train()

            for (x,) in train_loader:
                x = x.to(self.device)
                z = self.net(x)

                dist = torch.sum((z - self.c) ** 2, dim=1)

                if self.objective == "one-class":
                    loss = dist.mean()

                else:  # soft-boundary
                    loss = self.R**2 + (1 / self.nu) * torch.mean(torch.max(torch.zeros_like(dist), dist - self.R**2))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * x.size(0)
                n += x.size(0)

            if self.objective == "soft-boundary":
                self.R = torch.quantile(torch.sqrt(dist.detach()), 1 - self.nu)

            if epoch % print_every == 0:
                print(f"Epoch {epoch}/{n_epochs} | Loss: {epoch_loss / n:.6f}")

    # ------------------------------------------------
    # Score samples (distance)
    # ------------------------------------------------
    def score(self, X):
        self.net.eval()
        with torch.no_grad():
            X = torch.tensor(X, dtype=torch.float32).to(self.device)
            Z = self.net(X)
            dist = torch.sum((Z - self.c) ** 2, dim=1)
        return dist.cpu().numpy()
