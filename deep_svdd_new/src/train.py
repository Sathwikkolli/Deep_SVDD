# File: train.py

import torch
from pathlib import Path
import numpy as np

from deepsvdd import DeepSVDDNet
from utilsdata import load_embeddings, make_dataloader
from utilssvdd import (
    init_center_c,
    train_one_epoch,
    update_radius,
    compute_scores,
    compute_eer,
    plot_histograms,
    plot_roc_curve,
    plot_umap,
    ensure_dir,
)


# =============================
# CONFIG
# =============================
ROOT = Path("/home/ksathwik/projects/deep_svdd_new")
EMB_DIR = ROOT / "embeddings"       # ← embeddings from extraction.py
RESULTS_DIR = ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
MODELS_DIR = RESULTS_DIR / "models"

ensure_dir(RESULTS_DIR)
ensure_dir(PLOTS_DIR)
ensure_dir(MODELS_DIR)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Using device: {DEVICE}")


def run_experiment():
    name = "1024_512"
    print(f"\n========== EXPERIMENT: {name} ==========")

    # -------------------------
    # Load embeddings
    # -------------------------
    train_real, eval_real, eval_spoof = load_embeddings(EMB_DIR)
    print(f"[INFO] Train Real: {train_real.shape} | Eval Real: {eval_real.shape} | Eval Spoof: {eval_spoof.shape}")

    train_loader = make_dataloader(train_real, batch_size=64, shuffle=True)

    # -------------------------
    # Model + Optimizer
    # -------------------------
    net = DeepSVDDNet(input_dim=1024, hidden=[], rep_dim=512).to(DEVICE)

    nu = 0.1
    lr = 1e-4
    weight_decay = 1e-4
    epochs = 80

    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

    print("[INFO] Initializing center c...")
    c = init_center_c(net, train_loader, DEVICE)

    R = torch.tensor(0.0, device=DEVICE)

    # -------------------------
    # Train
    # -------------------------
    print("[INFO] Training Deep-SVDD (1024 → 512)...")
    for epoch in range(1, epochs + 1):
        epoch_loss, dist_all = train_one_epoch(net, train_loader, optimizer, c, R, nu, DEVICE)

        if epoch > 5:  # update after warm-up
            R = update_radius(R, dist_all, nu)

        print(f"Epoch {epoch}/{epochs} | Loss={epoch_loss:.4f} | R={R.item():.4f}")

    # -------------------------
    # Save Model
    # -------------------------
    model_path = MODELS_DIR / f"deepsvdd_{name}.pt"
    torch.save({"state_dict": net.state_dict(), "c": c, "R": R, "nu": nu}, model_path)
    print(f"[SAVE] Model stored at → {model_path}")

    # -------------------------
    # Evaluate
    # -------------------------
    real_scores = compute_scores(net, c, R, eval_real, DEVICE)
    fake_scores = compute_scores(net, c, R, eval_spoof, DEVICE)

    eer, fpr, tpr = compute_eer(real_scores, fake_scores)
    print(f"\n🔥 EER (1024→512) = {eer:.4f}\n")

    # -------------------------
    # Visualizations
    # -------------------------
    PLOTS_DIR.mkdir(exist_ok=True, parents=True)

    # Histogram
    plot_histograms(real_scores, fake_scores, PLOTS_DIR / "hist_1024_512.png", "Score Histogram (1024→512)")

    # ROC
    plot_roc_curve(fpr, tpr, PLOTS_DIR / "roc_1024_512.png", "ROC Curve (1024→512)")

    # UMAP Raw 1024
    X_raw = np.vstack([eval_real, eval_spoof])
    labels_raw = [0]*len(eval_real) + [1]*len(eval_spoof)
    plot_umap(X_raw, labels_raw, PLOTS_DIR / "umap_raw_1024.png", "UMAP Raw (1024)")

    # UMAP Latent 512
    net.eval()
    with torch.no_grad():
        Zr = net(torch.tensor(eval_real).float().to(DEVICE)).cpu().numpy()
        Zs = net(torch.tensor(eval_spoof).float().to(DEVICE)).cpu().numpy()
    Z = np.vstack([Zr, Zs])
    plot_umap(Z, labels_raw, PLOTS_DIR / "umap_latent_512.png", "UMAP Latent (512)")

    return eer


if __name__ == "__main__":
    run_experiment()
