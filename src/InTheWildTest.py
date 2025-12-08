import os
import numpy as np
import torch
from deep_svdd import DeepSVDD
from utils_svdd import compute_scores, compute_eer
from utils_data import create_eval_loader
from visualization import visualize
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.metrics import roc_curve

MODEL_PATH = "/home/ksathwik/projects/deep_svdd/models/deep_svdd_trump.pt"

ITW_EMB = "/home/ksathwik/projects/deep_svdd/embeddings_itw_trump/itw_trump_emb.npy"
ITW_LABELS = "/home/ksathwik/projects/deep_svdd/embeddings_itw_trump/itw_trump_labels.npy"

SAVE_DIR = "/home/ksathwik/projects/deep_svdd/itw_trump_eval"
os.makedirs(SAVE_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------
# LOAD MODEL + SCALER
# --------------------------------------------------
state = torch.load(MODEL_PATH, map_location=DEVICE)
scaler: StandardScaler = state["scaler"]

model = DeepSVDD(
    objective=state["objective"],
    nu=state["nu"],
    input_dim=6144,  # First 6 layers concatenated
    hidden_dims=(2048, 512),
    rep_dim=128,
    device=DEVICE,
)

model.net.load_state_dict(state["net_state_dict"])
model.c = state["center_c"].to(DEVICE)
model.R = state["radius_R"].to(DEVICE)

print("\n[LOADED] Deep SVDD model + scaler")


# --------------------------------------------------
# LOAD ITW EMBEDDINGS
# --------------------------------------------------
X = np.load(ITW_EMB)
y = np.load(ITW_LABELS)

print(f"[ITW-TRUMP] Loaded embeddings: {X.shape}")


# --------------------------------------------------
# SCALE
# --------------------------------------------------
X_s = scaler.transform(X)
loader = create_eval_loader(X_s, batch_size=256)


# --------------------------------------------------
# COMPUTE SCORES
# --------------------------------------------------
scores = compute_scores(model, loader)

# Save numpy files
np.save(os.path.join(SAVE_DIR, "itw_trump_scores.npy"), scores)
np.save(os.path.join(SAVE_DIR, "itw_trump_labels.npy"), y)

print(f"[SCORES] Saved score arrays to: {SAVE_DIR}")


# --------------------------------------------------
# Save CSV for easy viewing
# --------------------------------------------------
csv_path = os.path.join(SAVE_DIR, "itw_trump_scores.csv")
df = pd.DataFrame({"score": scores, "label": y})
df.to_csv(csv_path, index=False)
print(f"[CSV] Human-readable scores saved → {csv_path}")


# --------------------------------------------------
# SPLIT REAL / SPOOF
# --------------------------------------------------
spoof_scores = scores[y == 1]
real_scores = scores[y == 0]

print(f"\n[COUNT] Real: {len(real_scores)} | Spoof: {len(spoof_scores)}")
print(f"[STATS] Real Mean={real_scores.mean():.4f}, Spoof Mean={spoof_scores.mean():.4f}")

# Check for NaN in real_scores and spoof_scores
if np.any(np.isnan(real_scores)):
    print(f"[ERROR] NaN detected in real_scores. Fixing...")
    real_scores = np.nan_to_num(real_scores)  # Replace NaNs with zeros

if np.any(np.isnan(spoof_scores)):
    print(f"[ERROR] NaN detected in spoof_scores. Fixing...")
    spoof_scores = np.nan_to_num(spoof_scores)  # Replace NaNs with zeros


# --------------------------------------------------
# EER
# --------------------------------------------------
eer, eer_thr = compute_eer(real_scores, spoof_scores)

print("\n========== ITW TRUMP EER ==========")
print(f"EER:          {eer:.6f}")
print(f"EER threshold: {eer_thr:.6f}")

# FPR@EER / TPR@EER
# recompute with thresholds
y_full = np.concatenate([np.zeros_like(real_scores), np.ones_like(spoof_scores)])
scores_full = np.concatenate([real_scores, spoof_scores])
fpr, tpr, th = roc_curve(y_full, scores_full)

idx = np.nanargmin(np.abs(fpr - (1 - tpr)))
print(f"FPR@EER:      {fpr[idx]:.6f}")
print(f"TPR@EER:      {tpr[idx]:.6f}")
print("===================================\n")


# --------------------------------------------------
# VISUALIZATION
# --------------------------------------------------
visualize(real_scores, spoof_scores, SAVE_DIR)
print("\n[VIS] Saved ITW plots.")
