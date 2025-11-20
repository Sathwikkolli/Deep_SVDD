# src/utils_svdd.py
import numpy as np
import torch


# ----------------------------------------------------
# Compute distances from Deep SVDD model
# ----------------------------------------------------
def compute_scores(model, center_c, X, device):
    model.net.eval()

    with torch.no_grad():
        X = torch.tensor(X, dtype=torch.float32).to(device)
        Z = model.net(X)
        dist = torch.sum((Z - center_c) ** 2, dim=1)

    return dist.cpu().numpy()


# ----------------------------------------------------
# Threshold from REAL distribution
# ----------------------------------------------------
def compute_threshold_from_real(dist_real, percentile=95):
    """
    Compute simple percentile threshold from real scores.
    """
    return np.percentile(dist_real, percentile)


# ----------------------------------------------------
# Fixed threshold evaluation (TPR/FPR/ACC)
# ----------------------------------------------------
def evaluate_fixed_threshold(dist_real, dist_spoof, threshold):
    tp = (dist_real <= threshold).sum()
    fn = (dist_real > threshold).sum()

    tn = (dist_spoof > threshold).sum()
    fp = (dist_spoof <= threshold).sum()

    tpr = tp / (tp + fn + 1e-9)
    fpr = fp / (fp + tn + 1e-9)
    acc = (tp + tn) / (tp + tn + fp + fn + 1e-9)

    return dict(TPR=tpr, FPR=fpr, ACC=acc)


# ----------------------------------------------------
# Compute EER (Equal Error Rate)
# ----------------------------------------------------
def compute_eer(dist_real, dist_spoof):
    scores = np.concatenate([dist_real, dist_spoof])
    labels = np.concatenate([np.zeros_like(dist_real), np.ones_like(dist_spoof)])

    thresholds = np.sort(scores)

    fprs = []
    fnrs = []

    for t in thresholds:
        fp = ((dist_spoof <= t)).sum()
        tn = ((dist_spoof > t)).sum()
        fn = ((dist_real > t)).sum()
        tp = ((dist_real <= t)).sum()

        fpr = fp / (fp + tn + 1e-9)
        fnr = fn / (fn + tp + 1e-9)

        fprs.append(fpr)
        fnrs.append(fnr)

    fprs = np.array(fprs)
    fnrs = np.array(fnrs)
    diff = np.abs(fprs - fnrs)

    idx = np.argmin(diff)
    eer = (fprs[idx] + fnrs[idx]) / 2
    eer_threshold = thresholds[idx]

    return float(eer), float(eer_threshold), float(fprs[idx]), float(fnrs[idx])
