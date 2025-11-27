# src/utils_svdd.py

import numpy as np
import torch
from sklearn.metrics import roc_curve


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

            dist = torch.sum((z - model.c) ** 2, dim=1)
            epoch_dists.append(dist.detach())

            if model.objective == "one-class":
                loss = torch.mean(dist)
            else:
                scores = dist - model.R**2
                loss = model.R**2 + (1/model.nu) * torch.mean(torch.clamp(scores, min=0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x_batch.size(0)
            n_samples += x_batch.size(0)

        if model.objective == "soft-boundary":
            all_dist = torch.cat(epoch_dists)
            model.R = torch.quantile(torch.sqrt(all_dist), 1 - model.nu).detach()

        print(f"[SVDD] Ep {ep}/{epochs} Loss={running_loss/n_samples:.6f} R={model.R:.4f}")

    return model


@torch.no_grad()
def compute_scores(model, loader):
    device = model.device
    net = model.net
    net.eval()

    all_scores = []
    for (x_batch,) in loader:
        x_batch = x_batch.to(device)
        z = net(x_batch)
        dist = torch.sum((z - model.c)**2, dim=1)
        all_scores.append(dist.cpu().numpy())

    return np.concatenate(all_scores)


def compute_threshold_from_real(dist_real, percentile=95):
    return np.percentile(dist_real, percentile)


def evaluate_fixed_threshold(dist_real, dist_spoof, threshold):
    real_ok = dist_real <= threshold
    spoof_ok = dist_spoof <= threshold

    tp = real_ok.sum()
    fn = (~real_ok).sum()
    fp = spoof_ok.sum()
    tn = (~spoof_ok).sum()

    tpr = tp / (tp+fn+1e-12)
    fpr = fp / (fp+tn+1e-12)
    acc = (tp+tn) / (tp+tn+fp+fn)

    return float(tpr), float(fpr), float(acc)


def compute_eer(dist_real, dist_spoof):
    y = np.concatenate([np.zeros_like(dist_real), np.ones_like(dist_spoof)])
    scores = np.concatenate([dist_real, dist_spoof])

    fpr, tpr, thr = roc_curve(y, scores)
    fnr = 1 - tpr

    idx = np.argmin(np.abs(fnr - fpr))
    eer = (fnr[idx] + fpr[idx]) / 2
    eer_thr = thr[idx]
    return float(eer), float(eer_thr)
