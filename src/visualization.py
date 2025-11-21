# visualization.py (one-shot version - no model loading needed)

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import umap
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix, DetCurveDisplay
from sklearn.decomposition import PCA

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_DIR = "plots"
os.makedirs(SAVE_DIR, exist_ok=True)


def compute_umap(X):
    reducer = umap.UMAP(n_neighbors=20, min_dist=0.1)
    return reducer.fit_transform(X)


def visualize(svdd, X_eval_real, X_eval_spoof):

    print("\n[INFO] Generating visualizations...")

    # -------------------------------
    # Compute distances
    # -------------------------------
    dist_real = svdd.score(X_eval_real)
    dist_spoof = svdd.score(X_eval_spoof)

    # -------------------------------
    # Compute latent embeddings
    # -------------------------------
    with torch.no_grad():
        Z_real = svdd.net(torch.tensor(X_eval_real).float().to(DEVICE)).cpu().numpy()
        Z_spoof = svdd.net(torch.tensor(X_eval_spoof).float().to(DEVICE)).cpu().numpy()

    # -------------------------------
    # Score Distribution
    # -------------------------------
    plt.figure(figsize=(7,4))
    sns.kdeplot(dist_real, fill=True, label="Real")
    sns.kdeplot(dist_spoof, fill=True, label="Spoof")
    plt.legend()
    plt.title("Score Distribution")
    plt.savefig(f"{SAVE_DIR}/score_distribution.png", dpi=300)
    plt.close()

    # -------------------------------
    # ROC Curve
    # -------------------------------
    y_true = np.concatenate([np.zeros_like(dist_real), np.ones_like(dist_spoof)])
    y_scores = np.concatenate([dist_real, dist_spoof])

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(7,5))
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.4f}")
    plt.plot([0,1],[0,1],'--')
    plt.legend()
    plt.title("ROC Curve")
    plt.savefig(f"{SAVE_DIR}/roc_curve.png", dpi=300)
    plt.close()

    # -------------------------------
    # DET Curve
    # -------------------------------
    fig, ax = plt.subplots(figsize=(7,5))
    DetCurveDisplay.from_predictions(y_true, y_scores, ax=ax)
    plt.title("DET Curve")
    plt.savefig(f"{SAVE_DIR}/det_curve.png", dpi=300)
    plt.close()

    # -------------------------------
    # UMAP Raw
    # -------------------------------
    X = np.concatenate([X_eval_real, X_eval_spoof])
    labels = np.array([0]*len(X_eval_real) + [1]*len(X_eval_spoof))

    Z_raw = compute_umap(X)

    plt.figure(figsize=(8,5))
    plt.scatter(Z_raw[labels==0,0], Z_raw[labels==0,1], s=8, label="Real")
    plt.scatter(Z_raw[labels==1,0], Z_raw[labels==1,1], s=8, label="Spoof")
    plt.legend()
    plt.title("UMAP Raw WavLM Embeddings")
    plt.savefig(f"{SAVE_DIR}/umap_raw.png", dpi=300)
    plt.close()

    # -------------------------------
    # UMAP Latent
    # -------------------------------
    Z_latent = np.concatenate([Z_real, Z_spoof])
    labels_lat = np.array([0]*len(Z_real) + [1]*len(Z_spoof))

    Z_lat = compute_umap(Z_latent)

    plt.figure(figsize=(8,5))
    plt.scatter(Z_lat[labels_lat==0,0], Z_lat[labels_lat==0,1], s=8, label="Real")
    plt.scatter(Z_lat[labels_lat==1,0], Z_lat[labels_lat==1,1], s=8, label="Spoof")
    plt.legend()
    plt.title("UMAP Latent Space")
    plt.savefig(f"{SAVE_DIR}/umap_latent.png", dpi=300)
    plt.close()

    print("\n[✔] All visualizations saved to:", SAVE_DIR)
    