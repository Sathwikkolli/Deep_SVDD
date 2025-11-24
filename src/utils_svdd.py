# src/utils_svdd.py

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_curve


# ----------------------------------------------------
# Autoencoder pretrain (UPDATED — device argument added)
# ----------------------------------------------------
def pretrain_autoencoder(ae, train_loader, epochs, lr, weight_decay, device="cuda"):
    device = torch.device(device)
    ae = ae.to(device)
    ae.train()

    optimizer = torch.optim.Adam(ae.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    for ep in range(1, epochs + 1):
        running_loss = 0.0
        n = 0

        for (x_batch,) in train_loader:
            x_batch = x_batch.to(device)

            optimizer.zero_grad()
            x_rec = ae(x_batch)
            loss = criterion(x_rec, x_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x_batch.size(0)
            n += x_batch.size(0)

        ep_loss = running_loss / n
        print(f"[AE] Epoch {ep}/{epochs} Loss={ep_loss:.6f}")

    return ae


# ----------------------------------------------------
# Deep SVDD training
# ----------------------------------------------------
def train_deep_svdd(model, train_loader, epochs, lr, weight_decay):
    device = model.device
    net = model.net
    net.train()

    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

    for ep in range(1, epochs + 1):
        running_loss = 0.0
        n_samples = 0
        epoch_dists = []

        for (x_batch,) in train_loader:
            x_batch = x_batch.to(device)
            z = net(x_batch)

            # squared distance to center
            dist = torch.sum((z - model.c) ** 2, dim=1)
            epoch_dists.append(dist.detach())

            if model.objective == "one-class":
                loss = torch.mean(dist)
            else:  # soft-boundary
                scores = dist - (model.R ** 2)
                loss = (model.R ** 2) + (1.0 / model.nu) * torch.mean(
                    torch.clamp(scores, min=0.0)
                )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x_batch.size(0)
            n_samples += x_batch.size(0)

        # update R for soft-boundary
        if model.objective == "soft-boundary":
            all_dist = torch.cat(epoch_dists)
            model.R = torch.quantile(torch.sqrt(all_dist), 1.0 - model.nu).detach()

        ep_loss = running_loss / n_samples
        print(f"[SVDD] Ep {ep}/{epochs} Loss={ep_loss:.6f} R={model.R.item():.4f}")

    return model


# ----------------------------------------------------
# Compute anomaly scores
# ----------------------------------------------------
@torch.no_grad()
def compute_scores(model, loader):
    device = model.device
    net = model.net
    net.eval()

    all_scores = []

    for (x_batch,) in loader:
        x_batch = x_batch.to(device)
        z = net(x_batch)
        dist = torch.sum((z - model.c) ** 2, dim=1)
        all_scores.append(dist.cpu().numpy())

    return np.concatenate(all_scores, axis=0)


# ----------------------------------------------------
# Thresholds and metrics
# ----------------------------------------------------
def compute_threshold_from_real(dist_real, percentile=95):
    return np.percentile(dist_real, percentile)


def evaluate_fixed_threshold(dist_real, dist_spoof, threshold):
    real_pred_real = dist_real <= threshold
    real_pred_spoof = dist_spoof <= threshold

    tp = real_pred_real.sum()
    fn = (~real_pred_real).sum()
    fp = real_pred_spoof.sum()
    tn = (~real_pred_spoof).sum()

    tpr = tp / (tp + fn + 1e-12)
    fpr = fp / (fp + tn + 1e-12)
    acc = (tp + tn) / (tp + tn + fp + fn + 1e-12)

    return float(tpr), float(fpr), float(acc)


def compute_eer(dist_real, dist_spoof):
    y = np.concatenate(
        [np.zeros_like(dist_real, dtype=int), np.ones_like(dist_spoof, dtype=int)]
    )
    scores = np.concatenate([dist_real, dist_spoof])

    fpr, tpr, thresholds = roc_curve(y, scores)
    fnr = 1 - tpr

    idx = np.nanargmin(np.abs(fnr - fpr))
    eer = (fnr[idx] + fpr[idx]) / 2.0
    eer_thr = thresholds[idx]

    return float(eer), float(eer_thr)
