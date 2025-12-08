import os, numpy as np, pandas as pd
from pathlib import Path

ROOT = Path("/home/ksathwik/projects/deep_svdd_new")
DATA_ROOT = Path("/nfs/turbo/umd-hafiz/issf_server_data/famousfigures/Donald_Trump")   # REAL DATA
EMB_DIR = ROOT / "embeddings"
EXPORT_DIR = ROOT / "real_like_spoofs"
EXPORT_DIR.mkdir(exist_ok=True)

# -------- LOAD EMBEDDINGS --------
Z_real  = np.load(EMB_DIR/"eval_real_1024.npy")
Z_spoof = np.load(EMB_DIR/"eval_spoof_1024.npy")

# -------- FIND SPOOFS THAT LOOK REAL --------
center = Z_real.mean(axis=0)
d = np.linalg.norm(Z_spoof - center, axis=1)
idx = np.argsort(d)[:15]

# -------- LOAD PROTOCOL (for filenames only) --------
csv = pd.read_csv(ROOT/"oc_protocol_eval1000.csv")
eval_spoof = csv[csv.split=="eval"]                   # no path used
eval_spoof = eval_spoof.reset_index(drop=True)

# -------- MAP FILENAME TO REAL PATH ON SYSTEM --------
def locate_file(filename):
    for root,_,files in os.walk(DATA_ROOT):
        if filename in files:
            return os.path.join(root, filename)
    return None

print("\n🔥 MOST REAL-LIKE SPOOF FILES 🔽\n")

files = []
for i in idx:
    fname = os.path.basename(eval_spoof.iloc[i]["audiofilepath"])
    real_path = locate_file(fname)
    files.append(real_path)
    print(f"{len(files)}) {real_path}")

# -------- OPTIONAL EXPORT --------
import shutil
for p in files:
    if p:
        shutil.copy(p, EXPORT_DIR)

print("\n📁 Exported to", EXPORT_DIR)
