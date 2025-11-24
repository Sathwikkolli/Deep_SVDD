# src/utils_data.py

import os
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import torch
from sklearn.preprocessing import StandardScaler


# ----------------------------------------------------
# Load *.npy embeddings
# ----------------------------------------------------
def load_embeddings(emb_dir):
    emb_dir = os.path.abspath(emb_dir)

    X_train = np.load(os.path.join(emb_dir, "train_real.npy"))
    X_eval_real = np.load(os.path.join(emb_dir, "eval_real.npy"))
    X_eval_spoof = np.load(os.path.join(emb_dir, "eval_spoof.npy"))

    return X_train, X_eval_real, X_eval_spoof


# ----------------------------------------------------
# Scaling (fit only on TRAIN)
# ----------------------------------------------------
def scale_embeddings(X_train, X_eval_real, X_eval_spoof):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_eval_real_s = scaler.transform(X_eval_real)
    X_eval_spoof_s = scaler.transform(X_eval_spoof)
    return X_train_s, X_eval_real_s, X_eval_spoof_s, scaler


# ----------------------------------------------------
# DataLoaders
# ----------------------------------------------------
def create_train_loader(X, batch_size=128, device=None):
    # keep tensors on CPU; move to device in train loop
    tensor = torch.from_numpy(X).float()
    ds = TensorDataset(tensor)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)
    return loader


def create_eval_loader(X, batch_size=256, device=None):
    tensor = torch.from_numpy(X).float()
    ds = TensorDataset(tensor)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False)
    return loader
