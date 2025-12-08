# File: extraction.py

import os
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torchaudio
import torch.nn as nn
from s3prl.nn import S3PRLUpstream


# =============================
# CONFIGURATION
# =============================
PROTOCOL = "/home/ksathwik/projects/deep_svdd_new/oc_protocol_eval1000.csv"
DATA_ROOT = Path("/nfs/turbo/umd-hafiz/issf_server_data/famousfigures/Donald_Trump")
EMB_DIR = Path("/home/ksathwik/projects/deep_svdd_new/embeddings")

EMB_DIR.mkdir(exist_ok=True, parents=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Using device: {DEVICE}")


# =============================
# SSL MODEL WRAPPER (WavLM-Large)
# Last transformer layer + weighted pooling → 1024-dim
# =============================
class SSLModel(nn.Module):
    def __init__(self, model_name="wavlm_large", device="cuda"):
        super().__init__()
        self.device = device
        print(f"[INFO] Loading S3PRL upstream model: {model_name}...")
        self.model = S3PRLUpstream(model_name).to(self.device)
        self.model.eval()

        # Attention pooling to get weighted average over time
        self.att = nn.Sequential(
            nn.Linear(1024, 256),
            nn.Tanh(),
            nn.Linear(256, 1),
        ).to(self.device)

    @torch.no_grad()
    def extract_last_layer_weighted(self, waveform: torch.Tensor) -> np.ndarray:
        """
        Takes a mono waveform tensor (T,) or (1,T) at 16 kHz.
        Returns a 1024-dim numpy embedding from last transformer layer using
        attention-weighted pooling over time.
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)  # (1, T)

        waveform = waveform.to(self.device)
        wav_len = torch.LongTensor([waveform.size(1)]).to(self.device)

        # S3PRL upstream forward
        all_hs, _ = self.model(waveform, wav_len)  # list of layer outputs

        # Take last transformer layer
        last = all_hs[-1]  # shape: (batch, time, dim) or (time, batch, dim) tuple

        if isinstance(last, tuple):
            # Some upstreams return (hidden, length)
            last = last[0].permute(1, 0, 2)  # (B, T, D)
        # else: assume already (B, T, D)

        # Attention pooling
        # last: (1, T, 1024)
        weights = self.att(last)           # (1, T, 1)
        weights = torch.softmax(weights, dim=1)
        pooled = torch.sum(weights * last, dim=1)  # (1, 1024)

        return pooled.squeeze(0).cpu().numpy()     # (1024,)


# =============================
# HELPERS
# =============================
def build_index(root: Path):
    """
    Build a filename -> [full_paths] index for .wav files under DATA_ROOT.
    """
    index = {}
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith(".wav"):
                fp = str(Path(dirpath) / f)
                index.setdefault(f, []).append(fp)
    return index


def load_and_process_audio(path: str):
    """
    Load audio as mono 16 kHz waveform.
    """
    try:
        wav, sr = torchaudio.load(path)

        # Mix down to mono if needed
        if wav.ndim > 1 and wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)

        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)

        wav = wav.squeeze()
        return wav, None
    except Exception as e:
        return None, str(e)


# =============================
# MAIN
# =============================
def main():
    df = pd.read_csv(PROTOCOL)
    print(f"[INFO] Loaded protocol: {len(df)} entries")

    file_index = build_index(DATA_ROOT)
    print(f"[INFO] Built file index with {len(file_index)} unique filenames")

    ssl_model = SSLModel(model_name="wavlm_large", device=DEVICE)

    train_real_embs = []
    eval_real_embs = []
    eval_spoof_embs = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting 1024-dim embeddings"):
        proto_path = row["audiofilepath"]
        split = row["split"].strip()

        filename = os.path.basename(proto_path)
        candidates = file_index.get(filename, [])

        if not candidates:
            continue

        full_path = None
        label = None

        # Your folder-based logic: prefer bonafide
        for c in candidates:
            if "/-/" in c:
                full_path = c
                label = "bonafide"
                break

        if full_path is None:
            full_path = candidates[0]
            label = "spoof"

        wav_tensor, err = load_and_process_audio(full_path)
        if wav_tensor is None:
            print(f"[ERROR] {full_path}: {err}")
            continue

        try:
            emb = ssl_model.extract_last_layer_weighted(wav_tensor)

            if split == "train" and label == "bonafide":
                train_real_embs.append(emb)
            elif split == "eval" and label == "bonafide":
                eval_real_embs.append(emb)
            elif split == "eval" and label == "spoof":
                eval_spoof_embs.append(emb)

        except RuntimeError as e:
            print(f"[CUDA ERROR] {filename}: {e}")
            torch.cuda.empty_cache()
            continue

    # =============================
    # SAVE EMBEDDINGS
    # =============================
    print("\n[INFO] Saving 1024-dim embeddings...")

    if train_real_embs:
        np.save(EMB_DIR / "train_real_1024.npy", np.stack(train_real_embs))
    if eval_real_embs:
        np.save(EMB_DIR / "eval_real_1024.npy", np.stack(eval_real_embs))
    if eval_spoof_embs:
        np.save(EMB_DIR / "eval_spoof_1024.npy", np.stack(eval_spoof_embs))

    print("\n=== EXTRACTION FINISHED ===")
    if train_real_embs:
        print(f"Embedding dimension: {train_real_embs[0].shape}")
    print(f"Train Real: {len(train_real_embs)}")
    print(f"Eval Real : {len(eval_real_embs)}")
    print(f"Eval Spoof: {len(eval_spoof_embs)}")


if __name__ == "__main__":
    main()
