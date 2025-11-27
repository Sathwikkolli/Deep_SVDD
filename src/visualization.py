# src/visualization.py

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, DetCurveDisplay
from sklearn.decomposition import PCA
import umap
import torch

from utils_data import create_eval_loader   # REQUIRED for UMAP


# ----------------------------------------------------
# UTIL
# ----------------------------------------------------
def ensure(path):
    os.makedirs(path, exist_ok=True)


# ----------------------------------------------------
# HISTOGRAM, ROC, DET
# ----------------------------------------------------
def plot_hist(dist_real, dist_spoof, save_path):
    plt.figure()
    plt.hist(dist_real, bins=50, alpha=0.5, label="Real")
    plt.hist(dist_spoof, bins=50, alpha=0.5, label="Spoof")
    plt.xlabel("SVDD Distance")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_roc(dist_real, dist_spoof, save_path):
    y = np.concatenate([
        np.zeros_like(dist_real, dtype=int),
        np.ones_like(dist_spoof, dtype=int)
    ])
    scores = np.concatenate([dist_real, dist_spoof])

    fpr, tpr, _ = roc_curve(y, scores)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], "--")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_det(dist_real, dist_spoof, save_path):
    y = np.concatenate([
        np.zeros_like(dist_real, dtype=int),
        np.ones_like(dist_spoof, dtype=int)
    ])
    scores = np.concatenate([dist_real, dist_spoof])

    plt.figure()
    DetCurveDisplay.from_predictions(y, scores)
    plt.title("DET Curve")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def visualize(dist_real, dist_spoof, save_dir):
    ensure(save_dir)
    plot_hist(dist_real, dist_spoof, os.path.join(save_dir, "hist.png"))
    plot_roc(dist_real, dist_spoof, os.path.join(save_dir, "roc.png"))
    plot_det(dist_real, dist_spoof, os.path.join(save_dir, "det.png"))
    print(f"[VIS] Saved basic plots to {save_dir}")


# ----------------------------------------------------
# LATENT EXTRACTION (Deep-SVDD)
# ----------------------------------------------------
@torch.no_grad()
def extract_latent(model, loader, device="cuda"):
    device = torch.device(device)
    model.net.eval()

    latents = []
    for (x_batch,) in loader:
        x_batch = x_batch.to(device)
        z = model.net(x_batch)   # 128-d vector
        latents.append(z.cpu().numpy())

    return np.concatenate(latents, axis=0)


# ----------------------------------------------------
# UMAP SCATTER
# ----------------------------------------------------
def plot_umap(data, labels, title, save_path):
    reducer = umap.UMAP(
        n_neighbors=20,
        min_dist=0.1,
        metric="euclidean",
        random_state=42,
    )

    emb = reducer.fit_transform(data)

    plt.figure(figsize=(7, 6))
    plt.scatter(
        emb[:, 0],
        emb[:, 1],
        c=labels,
        cmap="coolwarm",
        s=8,
        alpha=0.85
    )
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# ----------------------------------------------------
# FULL UMAP PIPELINE
# ----------------------------------------------------
def visualize_umap(
    model,
    X_raw_real,
    X_raw_spoof,
    train_loader_latent,
    save_dir,
    device="cuda"
):
    ensure(save_dir)

    # =============================================
    # (1) RAW → PCA(128) → UMAP
    # =============================================
    X_raw = np.concatenate([X_raw_real, X_raw_spoof], axis=0)
    labels_raw = np.concatenate([
        np.zeros(len(X_raw_real)),
        np.ones(len(X_raw_spoof)),
    ])

    # PCA component count fix (must be ≤ samples)
    pca_dim = min(128, X_raw.shape[1], X_raw.shape[0] - 1)

    pca = PCA(n_components=pca_dim, random_state=42)
    X_pca = pca.fit_transform(X_raw)

    plot_umap(
        X_pca,
        labels_raw,
        f"UMAP Raw Embeddings ({X_raw.shape[1]} dims → PCA({pca_dim}) → UMAP)",
        os.path.join(save_dir, "umap_raw.png")
    )

    # =============================================
    # (2) LATENT SPACE (SVDD 128-d) → UMAP
    # =============================================
    real_loader = create_eval_loader(X_raw_real, batch_size=128)
    spoof_loader = create_eval_loader(X_raw_spoof, batch_size=128)

    Z_real = extract_latent(model, real_loader, device=device)
    Z_spoof = extract_latent(model, spoof_loader, device=device)

    Z_latent = np.concatenate([Z_real, Z_spoof], axis=0)
    labels_latent = np.concatenate([
        np.zeros(len(Z_real)),
        np.ones(len(Z_spoof)),
    ])

    plot_umap(
        Z_latent,
        labels_latent,
        "UMAP Deep-SVDD Latent Space (128 dims)",
        os.path.join(save_dir, "umap_latent.png")
    )

    print(f"[UMAP] Saved raw + latent UMAP plots to {save_dir}")
