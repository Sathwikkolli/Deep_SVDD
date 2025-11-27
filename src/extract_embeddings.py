import os
import torch
import torchaudio
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from s3prl.nn import S3PRLUpstream


# =============================
# CONFIGURATION
# =============================
PROTOCOL = "/home/ksathwik/projects/deep_svdd/oc_protocol_eval1000.csv"
DATA_ROOT = Path("/nfs/turbo/umd-hafiz/issf_server_data/famousfigures/Donald_Trump")
EMB_DIR = Path("/home/ksathwik/projects/deep_svdd/embeddings")

EMB_DIR.mkdir(exist_ok=True, parents=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")


# =============================
# SSL MODEL WRAPPER
# =============================
class SSLModel(torch.nn.Module):
    def __init__(self, model_name="wavlm_large", device="cuda"):
        super().__init__()
        self.device = device
        print(f"Loading S3PRL upstream model: {model_name}...")
        self.model = S3PRLUpstream(model_name).to(self.device)
        self.model.eval()

    def extract_first_6_layers(self, waveform):
        """
        Extract first 6 transformer layers from WavLM-Large.
        Each layer = (T, 1024)
        Output embedding = mean-pool of concat(6 layers) = 6144-dim
        """

        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        wav_len = torch.LongTensor([waveform.size(1)]).to(self.device)
        waveform = waveform.to(self.device)

        with torch.no_grad():
            output = self.model(waveform, wav_len)

            if isinstance(output, tuple):
                all_hs = output[0]
            elif isinstance(output, dict):
                all_hs = output["hidden_states"]
            else:
                all_hs = output

        # Extract transformer layers (skip CNN, idx=0)
        transformer_layers = all_hs[1:]   # Expect 24 layers

        # STRICT first 6 layers
        selected = transformer_layers[:6]   # layers 1 to 6

        # Concatenate along feature dimension → (1, T, 6144)
        concat_feats = torch.cat(selected, dim=2)

        # Mean over time → (1, 6144)
        embedding = concat_feats.mean(dim=1)

        return embedding.squeeze(0).cpu().numpy()


# =============================
# INITIALIZE MODEL
# =============================
ssl_model = SSLModel(model_name="wavlm_large", device=DEVICE)


# =============================
# HELPERS
# =============================
def build_index(root):
    index = {}
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith(".wav"):
                fp = str(Path(dirpath) / f)
                index.setdefault(f, []).append(fp)
    return index


def load_and_process_audio(path):
    try:
        wav, sr = torchaudio.load(path)

        if wav.ndim > 1 and wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)

        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)

        wav = wav.squeeze()
        return wav, None
    except Exception as e:
        return None, str(e)


# =============================
# MAIN EXTRACTION LOOP
# =============================
df = pd.read_csv(PROTOCOL)
print(f"Loaded protocol: {len(df)} entries")

file_index = build_index(DATA_ROOT)

train_real_embs = []
eval_real_embs = []
eval_spoof_embs = []

for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting FIRST 6 layers"):
    proto_path = row["audiofilepath"]
    split = row["split"].strip()

    filename = os.path.basename(proto_path)
    candidates = file_index.get(filename, [])

    if not candidates:
        continue

    full_path = None
    label = None

    # Prefer bonafide folder
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
        emb = ssl_model.extract_first_6_layers(wav_tensor)

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
print("\nSaving 6-layer embeddings...")

if train_real_embs:
    np.save(EMB_DIR / "train_real.npy", np.stack(train_real_embs))

if eval_real_embs:
    np.save(EMB_DIR / "eval_real.npy", np.stack(eval_real_embs))

if eval_spoof_embs:
    np.save(EMB_DIR / "eval_spoof.npy", np.stack(eval_spoof_embs))

print("\n=== FINISHED ===")
print(f"Embedding Dimension: {train_real_embs[0].shape if train_real_embs else 'N/A'}")
print(f"Train Real: {len(train_real_embs)}")
print(f"Eval Real : {len(eval_real_embs)}")
print(f"Eval Spoof: {len(eval_spoof_embs)}")
