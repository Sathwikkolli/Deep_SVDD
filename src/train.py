# src/train.py
import os
import numpy as np
import torch

from deep_svdd import DeepSVDD
from utils_data import load_embeddings, create_train_loader
from utils_svdd import compute_threshold_from_real, evaluate_fixed_threshold, compute_eer
from visualization import visualize   

# --------------------------
# Configuration
# --------------------------
EMB_DIR = "/home/ksathwik/projects/deep_svdd/embeddings"
SAVE_DIR = "/home/ksathwik/projects/deep_svdd/models"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
INPUT_DIM = 1024
REP_DIM = 128
OBJECTIVE = "soft-boundary"    # or "one-class"
NU = 0.05
LR = 1e-4
EPOCHS = 100
BATCH = 64
WEIGHT_DECAY = 1e-6
REAL_PERCENT = 95


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    print("Loading embeddings...")
    X_train, X_eval_real, X_eval_spoof = load_embeddings(EMB_DIR)

    train_loader = create_train_loader(X_train, batch_size=BATCH, shuffle=True)

    svdd = DeepSVDD(
        input_dim=INPUT_DIM,
        rep_dim=REP_DIM,
        objective=OBJECTIVE,
        nu=NU,
        device=DEVICE
    )

    print("Initializing center...")
    svdd.init_center_c(train_loader)
    print("Center:", svdd.c.shape)

    print("Training...")
    svdd.train(train_loader, n_epochs=EPOCHS, lr=LR, weight_decay=WEIGHT_DECAY, print_every=1)

    print("\nComputing scores...")
    dist_real = svdd.score(X_eval_real)
    dist_spoof = svdd.score(X_eval_spoof)

    thr = compute_threshold_from_real(dist_real, REAL_PERCENT)
    metrics = evaluate_fixed_threshold(dist_real, dist_spoof, thr)

    eer, eer_thr, fpr_eer, fnr_eer = compute_eer(dist_real, dist_spoof)

    # print metrics
    print("\n=========== FIXED THRESHOLD METRICS ===========")
    print("Threshold:", thr)
    print("TPR:", metrics["TPR"])
    print("FPR:", metrics["FPR"])
    print("ACC:", metrics["ACC"])

    print("\n=========== EER METRICS ===========")
    print("EER:", eer)
    print("EER Threshold:", eer_thr)
    print("FPR@EER:", fpr_eer)
    print("FNR@EER:", fnr_eer)

    # Save model
    model_path = os.path.join(SAVE_DIR, "deep_svdd_trump.pt")
    torch.save(
        {
            "state_dict": svdd.net.state_dict(),
            "center_c": svdd.c.cpu().numpy(),
            "R": float(svdd.R.cpu().item()),
        },
        model_path
    )

    # Save results
    result_file = os.path.join(SAVE_DIR, "results.txt")
    with open(result_file, "w") as f:
        f.write("=========== FIXED THRESHOLD METRICS ============\n")
        f.write(f"Threshold: {thr}\n")
        f.write(f"TPR: {metrics['TPR']}\n")
        f.write(f"FPR: {metrics['FPR']}\n")
        f.write(f"ACC: {metrics['ACC']}\n\n")

        f.write("=========== EER METRICS ============\n")
        f.write(f"EER: {eer}\n")
        f.write(f"EER Threshold: {eer_thr}\n")
        f.write(f"FPR@EER: {fpr_eer}\n")
        f.write(f"FNR@EER: {fnr_eer}\n")

    print("\nSaved model to:", model_path)
    print("Saved results to:", result_file)

    # --------------------------
    # RUN VISUALIZATIONS HERE
    # --------------------------
    visualize(svdd, X_eval_real, X_eval_spoof)


if __name__ == "__main__":
    main()
