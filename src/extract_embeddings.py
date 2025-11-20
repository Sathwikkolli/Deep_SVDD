import os
import torch
import torchaudio
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from transformers import WavLMModel, Wav2Vec2FeatureExtractor

# =============================
# CONFIG PATHS
# =============================
PROTOCOL = "/home/ksathwik/projects/deep_svdd/oc_protocol_eval1000.csv"
DATA_ROOT = Path("/nfs/turbo/umd-hafiz/issf_server_data/famousfigures/Donald_Trump")
EMB_DIR = Path("/home/ksathwik/projects/deep_svdd/embeddings")

EMB_DIR.mkdir(exist_ok=True, parents=True)

# =============================
# LOAD WAVLM LARGE
# =============================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-large")
model = WavLMModel.from_pretrained("microsoft/wavlm-large").to(DEVICE)
model.eval()

# =============================
# BUILD FILENAME → PATH INDEX
# =============================
def build_index(root):
    index = {}
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith(".wav"):
                fp = str(Path(dirpath) / f)
                index.setdefault(f, []).append(fp)
    return index

# =============================
# LOAD PROTOCOL
# =============================
df = pd.read_csv(PROTOCOL)
print("Loaded protocol entries:", len(df))

index = build_index(DATA_ROOT)
print("Indexed audio files:", sum(len(v) for v in index.values()))

# =============================
# WAVLM EMBEDDING FUNCTION
# =============================
def extract_embedding(path):
    try:
        wav, sr = torchaudio.load(path)

        # Stereo → mono
        if wav.ndim > 1 and wav.shape[0] > 1:
            wav = wav.mean(dim=0)

        wav = wav.squeeze()

        # Resample to 16k
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)

        wav = wav.numpy()  # <-- IMPORTANT FIX

        inputs = feature_extractor(
            wav,
            sampling_rate=16000,
            return_tensors="pt"
        )

        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            out = model(**inputs).last_hidden_state  # [1, T, 1024]

        emb = out.mean(dim=1).squeeze(0).cpu().numpy()
        return emb, None

    except Exception as e:
        return None, str(e)

# =============================
# STORAGE
# =============================
train_real_embs = []
eval_real_embs = []
eval_spoof_embs = []

# =============================
# PROCESS EACH PROTOCOL ROW
# =============================
for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing protocol"):
    proto_path = row["audiofilepath"]
    split = row["split"].strip()
    filename = os.path.basename(proto_path)

    candidates = index.get(filename, [])
    if not candidates:
        print(f"[MISSING] {filename}")
        continue

    # Detect bonafide: folder named "-"
    full_path = None
    for c in candidates:
        if "/-/" in c:
            full_path = c
            label = "bonafide"
            break

    if full_path is None:
        full_path = candidates[0]
        label = "spoof"

    emb, err = extract_embedding(full_path)
    if emb is None:
        print(f"[ERROR] {full_path} {err}")
        continue

    # store final
    if split == "train" and label == "bonafide":
        train_real_embs.append(emb)

    elif split == "eval" and label == "bonafide":
        eval_real_embs.append(emb)

    elif split == "eval" and label == "spoof":
        eval_spoof_embs.append(emb)

# =============================
# SAVE EMBEDDINGS
# =============================
np.save(EMB_DIR / "train_real.npy", np.stack(train_real_embs))
np.save(EMB_DIR / "eval_real.npy", np.stack(eval_real_embs))
np.save(EMB_DIR / "eval_spoof.npy", np.stack(eval_spoof_embs))

print("\n=== DONE ===")
print("train_real:", len(train_real_embs))
print("eval_real :", len(eval_real_embs))
print("eval_spoof:", len(eval_spoof_embs))
