# src/utils_data.py
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


# ===========================
# Load embeddings from folder
# ===========================
def load_embeddings(emb_dir):
    X_train = np.load(os.path.join(emb_dir, "train_real.npy"))
    X_eval_real = np.load(os.path.join(emb_dir, "eval_real.npy"))
    X_eval_spoof = np.load(os.path.join(emb_dir, "eval_spoof.npy"))
    return X_train, X_eval_real, X_eval_spoof


# ===========================
# Create PyTorch DataLoader
# ===========================
def create_train_loader(X, batch_size=64, shuffle=True):
    tensor = torch.tensor(X, dtype=torch.float32)
    dataset = TensorDataset(tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
