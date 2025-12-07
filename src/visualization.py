import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, DetCurveDisplay
from sklearn.decomposition import PCA
import umap
import torch
from utils_data import create_eval_loader


# ======================================================
#  ✨ FIX: force save path to /models/results ALWAYS
# ======================================================
BASE_SAVE = os.path.join(os.getcwd(), "models", "results")

def ensure():
    os.makedirs(BASE_SAVE, exist_ok=True)
    return BASE_SAVE


# ======================================================
# HISTOGRAM + ROC + DET
# ======================================================
def visualize(dist_real, dist_spoof):

    save_dir = ensure()   # <--- folder is created correctly

    # ---------- HIST ----------
    plt.figure()
    plt.hist(dist_real, bins=60, alpha=0.55, label="Real")
    plt.hist(dist_spoof, bins=60, alpha=0.55, label="Spoof")
    plt.xlabel("SVDD Distance")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{save_dir}/hist.png")
    plt.close()

    # ---------- ROC ----------
    y = np.concatenate([np.zeros_like(dist_real), np.ones_like(dist_spoof)])
    scores = np.concatenate([dist_real, dist_spoof])
    fpr, tpr, _ = roc_curve(y, scores)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.4f}")
    plt.plot([0,1],[0,1],'--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{save_dir}/roc_curve.png")
    plt.close()

    # ---------- DET ----------
    plt.figure()
    DetCurveDisplay.from_predictions(y, scores)
    plt.title("DET Curve")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/det_curve.png")
    plt.close()

    print(f"[+] Histogram, ROC, DET saved →  {save_dir}")


# ======================================================
# LATENT EXTRACTION
# ======================================================
@torch.no_grad()
def extract_latent(model, loader, device="cuda"):
    model.net.eval()
    device = torch.device(device)

    latents = []
    for (x_batch,) in loader:
        x_batch = x_batch.to(device)
        z = model.net(x_batch)
        latents.append(z.cpu().numpy())

    return np.concatenate(latents)


# ======================================================
# UMAP PLOTTING
# ======================================================
def plot_umap(data, labels, name):

    save_dir = ensure()

    reducer = umap.UMAP(
        n_neighbors=20,
        min_dist=0.1,
        metric="euclidean",
        random_state=42,
    )

    emb = reducer.fit_transform(data)

    plt.figure(figsize=(6,6))
    plt.scatter(emb[:,0], emb[:,1], c=labels, cmap="coolwarm", s=10, alpha=0.85)
    plt.title(name)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/{name}.png")
    plt.close()


# ======================================================
# UMAP COMPLETE PIPELINE
# ======================================================
def visualize_umap(model, X_real, X_spoof, device="cuda"):

    save_dir = ensure()

    # ---------- RAW Embeddings → PCA → UMAP ----------
    X = np.concatenate([X_real, X_spoof])
    labels = np.concatenate([np.zeros(len(X_real)), np.ones(len(X_spoof))])

    pca_dim = min(128, X.shape[1], len(X)-1)
    X_pca = PCA(n_components=pca_dim).fit_transform(X)

    plot_umap(X_pca, labels, f"UMAP_RAW_PCA_{pca_dim}")


    # ---------- LATENT 128-D → UMAP ----------
    real_loader  = create_eval_loader(X_real,  batch_size=128)
    spoof_loader = create_eval_loader(X_spoof, batch_size=128)

    Z_real  = extract_latent(model, real_loader,  device=device)
    Z_spoof = extract_latent(model, spoof_loader, device=device)

    Z = np.concatenate([Z_real, Z_spoof])
    Z_labels = np.concatenate([np.zeros(len(Z_real)), np.ones(len(Z_spoof))])

    plot_umap(Z, Z_labels, "UMAP_LATENT_128")

    print(f"[+] UMAP plots saved →  {save_dir}")
