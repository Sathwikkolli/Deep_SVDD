# src/train.py

import os
import random
import numpy as np
import torch

from deep_svdd import DeepSVDD
from utils_data import (
    load_embeddings,
    scale_embeddings,
    create_train_loader,
    create_eval_loader,
)
from utils_svdd import (
    train_deep_svdd,
    compute_scores,
    compute_threshold_from_real,
    evaluate_fixed_threshold,
    compute_eer,
)
from visualization import visualize, visualize_umap


# ----------------------------------------------------
# CONFIG
# ----------------------------------------------------
EMB_DIR = "/home/ksathwik/projects/deep_svdd/embeddings"
SAVE_DIR = "/home/ksathwik/projects/deep_svdd/models"
PLOTS_DIR = os.path.join(SAVE_DIR, "plots")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# FIRST 6 LAYERS → 6 × 1024 = 6144
INPUT_DIM = 6144

# SVDD encoder architecture
HIDDEN_DIMS = (2048, 512)
REP_DIM = 128

OBJECTIVE = "soft-boundary"
NU = 0.05

SVDD_EPOCHS = 60
BATCH_SIZE = 128
LR_SVDD = 1e-4
WEIGHT_DECAY = 1e-6


# ----------------------------------------------------
# SEEDING
# ----------------------------------------------------
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ----------------------------------------------------
# MAIN PIPELINE
# ----------------------------------------------------
def main():

    seed_everything(42)
    os.makedirs(SAVE_DIR, exist_ok=True)

    print("=== LOAD EMBEDDINGS ===")
    X_train, X_eval_real, X_eval_spoof = load_embeddings(EMB_DIR)
    print(
        f"Train Real: {X_train.shape}, "
        f"Eval Real: {X_eval_real.shape}, "
        f"Eval Spoof: {X_eval_spoof.shape}"
    )

    print("\n=== SCALING ===")
    X_train_s, X_eval_real_s, X_eval_spoof_s, scaler = scale_embeddings(
        X_train, X_eval_real, X_eval_spoof
    )

    train_loader = create_train_loader(X_train_s, batch_size=BATCH_SIZE)
    eval_real_loader = create_eval_loader(X_eval_real_s, batch_size=BATCH_SIZE*2)
    eval_spoof_loader = create_eval_loader(X_eval_spoof_s, batch_size=BATCH_SIZE*2)

    print("\n=== INIT SVDD MODEL ===")
    model = DeepSVDD(
        objective=OBJECTIVE,
        nu=NU,
        input_dim=INPUT_DIM,
        hidden_dims=HIDDEN_DIMS,
        rep_dim=REP_DIM,
        device=DEVICE,
    )

    print("\n=== INIT CENTER c ===")
    model.init_center_c(train_loader)

    print("\n=== TRAIN SVDD ===")
    model = train_deep_svdd(
        model,
        train_loader,
        epochs=SVDD_EPOCHS,
        lr=LR_SVDD,
        weight_decay=WEIGHT_DECAY
    )

    print("\n=== EVALUATION ===")
    dist_real  = compute_scores(model, eval_real_loader)
    dist_spoof = compute_scores(model, eval_spoof_loader)

    thr = compute_threshold_from_real(dist_real, percentile=95)
    tpr, fpr, acc = evaluate_fixed_threshold(dist_real, dist_spoof, thr)

    print("\n--- FIXED THRESHOLD METRICS ---")
    print(f"Threshold: {thr:.6f}")
    print(f"TPR: {tpr:.6f}")
    print(f"FPR: {fpr:.6f}")
    print(f"ACC: {acc:.6f}")

    eer, eer_thr = compute_eer(dist_real, dist_spoof)

    print("\n--- EER METRICS ---")
    print(f"EER: {eer:.6f}")
    print(f"EER Threshold: {eer_thr:.6f}")

    # Save results file
    results_file = os.path.join(SAVE_DIR, "results.txt")
    with open(results_file, "w") as f:
        f.write("=== Deep SVDD Results ===\n\n")
        f.write(f"Fixed Threshold(95%):  {thr:.6f}\n")
        f.write(f"TPR: {tpr:.6f}\nFPR: {fpr:.6f}\nACC: {acc:.6f}\n\n")
        f.write(f"EER: {eer:.6f}\nEER Threshold: {eer_thr:.6f}\n")
    print(f"[SAVED] metrics → {results_file}")


    print("\n=== VISUALIZATION ===")
    visualize(dist_real, dist_spoof)


    print("\n=== UMAP VISUALIZATION ===")
    visualize_umap(model, X_eval_real_s, X_eval_spoof_s, device=DEVICE)


    # Save model
    model_path = os.path.join(SAVE_DIR, "deep_svdd_trump.pt")
    model.save_state(model_path, scaler=scaler)
    print(f"\n[SAVED MODEL] → {model_path}")


if __name__ == "__main__":
    main()
