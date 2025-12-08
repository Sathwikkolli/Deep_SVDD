# File: utilsdata.py

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class EmbeddingDataset(Dataset):
    """
    Simple dataset that wraps a numpy array of embeddings.
    """
    def __init__(self, arr: np.ndarray):
        self.arr = arr.astype(np.float32)

    def __len__(self):
        return self.arr.shape[0]

    def __getitem__(self, idx):
        return self.arr[idx]


def load_embeddings(emb_dir):
    """
    Loads train/eval real/fake embeddings from .npy files.
    """
    train_real = np.load(emb_dir / "train_real_1024.npy")
    eval_real = np.load(emb_dir / "eval_real_1024.npy")
    eval_spoof = np.load(emb_dir / "eval_spoof_1024.npy")
    return train_real, eval_real, eval_spoof


def make_dataloader(arr: np.ndarray, batch_size=64, shuffle=True):
    ds = EmbeddingDataset(arr)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)
    return dl
