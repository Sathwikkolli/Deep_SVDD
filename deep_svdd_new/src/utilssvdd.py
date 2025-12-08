# File: utilssvdd.py

import os
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import umap.umap_ as umap


# =============================
# CENTER INITIALIZATION
# =============================
def init_center_c(net, dataloader, device, eps=1e-6):
    """
    Initialize hypersphere center c as mean of network outputs on train data.
    """
    net.eval()
    n_samples = 0
    c = None

    with torch.no_grad():
        for x in dataloader:
            x = x.to(device)
            z = net(x)
            if c is None:
                c = torch.sum(z, dim=0)
            else:
                c += torch.sum(z, dim=0)
            n_samples += z.shape[0]

    c /= n_samples

    # Avoid numerical instability
    c[(abs(c) < eps) & (c < 0)] = -eps
    c[(abs(c) < eps) & (c > 0)] = eps

    return c


# =============================
# TRAINING ONE EPOCH (Deep-SVDD loss)
# =============================
def train_one_epoch(net, dataloader, optimizer, c, R, nu, device):
    """
    One epoch of Deep-SVDD training with soft-boundary objective:
    L = R^2 + (1/(nu * n)) * sum(max(0, dist_i - R^2))
    """
    net.train()
    epoch_loss = 0.0
    all_dists = []

    for x in dataloader:
        x = x.to(device)
        z = net(x)
        dist = torch.sum((z - c) ** 2, dim=1)

        # Hinge term
        hinge = torch.relu(dist - R ** 2)  # max(0, dist - R^2)

        loss = R ** 2 + (1.0 / (nu * x.shape[0])) * torch.sum(hinge)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * x.shape[0]
        all_dists.append(dist.detach().cpu())

    epoch_loss /= len(dataloader.dataset)
    all_dists = torch.cat(all_dists, dim=0)

    return epoch_loss, all_dists


# =============================
# UPDATE R FROM DISTANCES
# =============================
def update_radius(R, dist_all, nu):
    """
    Update radius R as (1 - nu)-quantile of distances.
    """
    new_R = torch.quantile(dist_all, 1 - nu)
    R.data = new_R
    return R


# =============================
# SCORING & EER
# =============================
def compute_scores(net, c, R, arr, device):
    """
    Compute Deep-SVDD scores = ||z - c||^2 - R^2 for each embedding.
    """
    net.eval()
    scores = []

    with torch.no_grad():
        for i in range(0, arr.shape[0], 256):
            x = torch.tensor(arr[i:i+256], dtype=torch.float32, device=device)
            z = net(x)
            dist = torch.sum((z - c) ** 2, dim=1)
            s = dist - R ** 2
            scores.append(s.cpu().numpy())

    return np.concatenate(scores, axis=0)


def compute_eer(real_scores, spoof_scores):
    """
    Compute EER from scores of real (0) and spoof (1).
    Higher score = more anomalous (more fake).
    """
    y_true = np.concatenate([np.zeros_like(real_scores), np.ones_like(spoof_scores)])
    y_scores = np.concatenate([real_scores, spoof_scores])

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    eer = fnr[np.argmin(np.abs(fnr - fpr))]
    return eer, fpr, tpr


# =============================
# VISUALIZATION HELPERS
# =============================
def ensure_dir(p: Path):
    p.mkdir(exist_ok=True, parents=True)


def plot_histograms(real_scores, spoof_scores, out_path: Path, title="Score Histogram"):
    plt.figure()
    plt.hist(real_scores, bins=50, alpha=0.5, label="Real")
    plt.hist(spoof_scores, bins=50, alpha=0.5, label="Spoof")
    plt.legend()
    plt.xlabel("Score")
    plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_roc_curve(fpr, tpr, out_path: Path, title="ROC Curve"):
    plt.figure()
    plt.plot(fpr, tpr, label="ROC")
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_umap(X, labels, out_path: Path, title="UMAP"):
    reducer = umap.UMAP(n_neighbors=25, min_dist=0.1, metric="euclidean")
    emb_2d = reducer.fit_transform(X)

    plt.figure()
    sc = plt.scatter(emb_2d[:, 0], emb_2d[:, 1], c=labels, cmap="coolwarm", s=5)
    plt.colorbar(sc)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
