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

# Ensure output directory exists
EMB_DIR.mkdir(exist_ok=True, parents=True)

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# =============================
# SSL MODEL CLASS (S3PRL WRAPPER)
# =============================
class SSLModel(torch.nn.Module):
    def __init__(self, model_name="wavlm_large", device="cuda"):
        super(SSLModel, self).__init__()
        self.device = device
        print(f"Loading S3PRL upstream model: {model_name}...")
        self.model = S3PRLUpstream(model_name).to(self.device)
        self.model.eval()

    def extract_concatenated_layers(self, waveform):
        """
        Extracts all transformer layers, concatenates them along the feature dimension,
        and performs mean pooling over time.
        """
        # s3prl expects a list of waveforms or a padded tensor
        # We ensure input is (1, time)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
            
        wav_len = torch.LongTensor([waveform.size(1)]).to(self.device)
        waveform = waveform.to(self.device)

        with torch.no_grad():
            output = self.model(waveform, wav_len)
            
            # FIX: Handle Tuple Return (hidden_states, lengths)
            if isinstance(output, tuple):
                all_hs = output[0]
            elif isinstance(output, dict):
                all_hs = output["hidden_states"]
            else:
                all_hs = output

        # We strictly want the 24 transformer layers (skipping index 0)
        # Index 0 is usually the CNN output (Projected Latent)
        transformer_layers = all_hs[1:] 
        
        # Verify we have 24 layers (WavLM Large standard)
        # If upstream structure differs, strictly take the last 24
        if len(transformer_layers) != 24:
            transformer_layers = all_hs[-24:]

        # Stack layers: List of (Batch, Time, Dim) -> (Batch, Layers, Time, Dim)
        # We want to concatenate features: (Batch, Time, Layers * Dim)
        
        # 1. Concatenate along the feature dimension (dim=2)
        # Each layer is [1, T, 1024]. Result is [1, T, 24576]
        concat_feats = torch.cat(transformer_layers, dim=2)
        
        # 2. Mean Pooling over time (dim=1)
        # Result is [1, 24576]
        embedding = concat_feats.mean(dim=1)

        return embedding.squeeze(0).cpu().numpy()

# =============================
# INITIALIZE MODEL
# =============================
# Initialize the wrapper class
ssl_model = SSLModel(model_name="wavlm_large", device=DEVICE)

# =============================
# HELPER FUNCTIONS
# =============================
def build_index(root):
    """Recursively finds all .wav files and indexes them by filename."""
    print(f"Indexing files in {root}...")
    index = {}
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith(".wav"):
                fp = str(Path(dirpath) / f)
                index.setdefault(f, []).append(fp)
    return index

def load_and_process_audio(path):
    """Loads audio, converts to mono, and resamples to 16kHz."""
    try:
        wav, sr = torchaudio.load(path)

        # Stereo -> Mono
        if wav.ndim > 1 and wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        
        # Resample to 16k if necessary
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
        
        # Ensure shape is (time,) for the SSLModel wrapper
        wav = wav.squeeze()
        
        return wav, None
    except Exception as e:
        return None, str(e)

# =============================
# MAIN EXECUTION
# =============================
# 1. Load Protocol
df = pd.read_csv(PROTOCOL)
print(f"Loaded protocol: {len(df)} entries")

# 2. Build File Index
file_index = build_index(DATA_ROOT)
print(f"Indexed audio files: {sum(len(v) for v in file_index.values())}")

# Storage
train_real_embs = []
eval_real_embs = []
eval_spoof_embs = []

# 3. Process Loop
for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting embeddings"):
    proto_path = row["audiofilepath"]
    split = row["split"].strip()
    
    # Handle filename matching
    filename = os.path.basename(proto_path)
    candidates = file_index.get(filename, [])

    if not candidates:
        # print(f"[MISSING] {filename}") # Uncomment to debug missing files
        continue

    # Logic to prioritize bonafide folders if duplicates exist
    full_path = None
    for c in candidates:
        if "/-/" in c:  # Assuming "-" folder indicates bonafide source
            full_path = c
            label = "bonafide"
            break
    
    if full_path is None:
        full_path = candidates[0]
        label = "spoof"

    # Load Audio
    wav_tensor, err = load_and_process_audio(full_path)
    if wav_tensor is None:
        print(f"[ERROR] {full_path}: {err}")
        continue

    # Extract Embedding (Concatenation of 24 layers)
    try:
        emb = ssl_model.extract_concatenated_layers(wav_tensor)
        
        # Store based on Split/Label
        if split == "train" and label == "bonafide":
            train_real_embs.append(emb)
        elif split == "eval" and label == "bonafide":
            eval_real_embs.append(emb)
        elif split == "eval" and label == "spoof":
            eval_spoof_embs.append(emb)
            
    except RuntimeError as e:
        print(f"[CUDA/Memory Error] {filename}: {e}")
        torch.cuda.empty_cache()
        continue

# =============================
# SAVE RESULTS
# =============================
print("\nSaving embeddings to disk...")

if train_real_embs:
    np.save(EMB_DIR / "train_real.npy", np.stack(train_real_embs))
else:
    print("Warning: No train_real embeddings found.")

if eval_real_embs:
    np.save(EMB_DIR / "eval_real.npy", np.stack(eval_real_embs))

if eval_spoof_embs:
    np.save(EMB_DIR / "eval_spoof.npy", np.stack(eval_spoof_embs))

print("\n=== FINISHED ===")
print(f"Dimensions per embedding: {train_real_embs[0].shape if train_real_embs else 'N/A'}")
print(f"Train Real: {len(train_real_embs)}")
print(f"Eval Real : {len(eval_real_embs)}")
print(f"Eval Spoof: {len(eval_spoof_embs)}")