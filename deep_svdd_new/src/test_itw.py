import os
from glob import glob
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import umap
from sklearn.metrics import roc_curve, accuracy_score
from s3prl.nn import S3PRLUpstream

from deepsvdd import DeepSVDDNet


# =========================
# CONFIG
# =========================
ROOT = Path("/home/ksathwik/projects/deep_svdd_new")

MODEL_PATH = ROOT / "results/models/deepsvdd_1024_512.pt"
ITW_WAV_DIR = Path("/nfs/turbo/umd-hafiz/issf_server_data/ds_wild/release_in_the_wild")
META_CSV = Path("/nfs/turbo/umd-hafiz/issf_server_data/ds_wild/protocols/meta.csv")

RESULTS_DIR = ROOT / "results"
PLOTS_DIR = ROOT / "plots"

RESULTS_DIR.mkdir(exist_ok=True, parents=True)
PLOTS_DIR.mkdir(exist_ok=True, parents=True)

BATCH_SIZE = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"[DEVICE] {DEVICE}")


# =========================
# 1) LOAD MODEL
# =========================
print("\n[1] Loading DeepSVDD model...")

ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
net = DeepSVDDNet(input_dim=1024, hidden=[], rep_dim=512).to(DEVICE)
net.load_state_dict(ckpt["state_dict"])
net.eval()

c = ckpt["c"].to(DEVICE)
R = ckpt["R"].to(DEVICE)
nu = ckpt["nu"]

print(f"    Loaded ✓ R={R.item():.6f}, nu={nu}")


# =========================
# 2) LOAD META & FILTER TRUMP
# =========================
print("\n[2] Loading ITW metadata...")

df = pd.read_csv(META_CSV)
df.columns = [c.strip().lower() for c in df.columns]

df_trump = df[df["speaker"].str.contains("donald", case=False)].reset_index(drop=True)

df_trump["y_true"] = df_trump["label"].replace({"bona-fide":0,"bonafide":0,"spoof":1})

print(f"    Donald Trump files: {len(df_trump)}")
print(df_trump["label"].value_counts())


# =========================
# 3) MAP FILE → PATHS
# =========================
all_wavs = glob(str(ITW_WAV_DIR/"*.wav"))

file_map = {os.path.basename(p):p for p in all_wavs}
df_trump["path"] = df_trump["file"].apply(lambda f:file_map.get(f,None))
df_trump = df_trump[df_trump["path"].notna()].reset_index(drop=True)

paths = df_trump["path"].tolist()
y_true = df_trump["y_true"].values
print(f"Resolved {len(paths)} ITW paths.")


# =========================
# 4) WAVLM EXTRACTOR
# =========================
print("\n[4] Loading WavLM-Large...")

ssl_model = S3PRLUpstream("wavlm_large").to(DEVICE)
ssl_model.eval()

@torch.no_grad()
def extract_wavlm_1024(path):
    y, _ = librosa.load(path, sr=16000)
    y = torch.tensor(y,device=DEVICE).unsqueeze(0)
    l = torch.LongTensor([y.shape[1]]).to(DEVICE)

    hs,_ = ssl_model(y,l)
    return hs[-1].mean(dim=1).cpu().numpy()  # (1024,)


# =========================
# 5) EXTRACT RAW FEATURES
# =========================
print("\n[5] Extracting 1024-d WavLM embeddings...")

emb=[]
from tqdm import tqdm
for i in tqdm(range(0,len(paths),BATCH_SIZE)):
    batch=[]
    for p in paths[i:i+BATCH_SIZE]:
        try: batch.append(extract_wavlm_1024(p))
        except: batch.append(np.zeros(1024))
    emb.append(np.vstack(batch))

X_raw=np.vstack(emb)
np.save(RESULTS_DIR/"trump_itw_raw_1024.npy",X_raw)
print(f"RAW Shape = {X_raw.shape}")


# =========================
# 6) SVDD LATENT + SCORES
# =========================
print("\n[6] Forward through DeepSVDD...")

Z=[]
with torch.no_grad():
    for i in tqdm(range(0,len(X_raw),BATCH_SIZE)):
        x=torch.tensor(X_raw[i:i+BATCH_SIZE],dtype=torch.float32,device=DEVICE)
        Z.append(net(x).cpu().numpy())

X_lat=np.vstack(Z)
np.save(RESULTS_DIR/"trump_itw_lat_512.npy",X_lat)

c_np=c.cpu().numpy()
R_val=R.item()

dist=np.linalg.norm(X_lat-c_np,axis=1)
margin=dist-R_val
y_pred=(margin>0).astype(int)


# =========================
# 7) METRICS
# =========================
print("\n[7] Evaluating...")

fpr,tpr,_=roc_curve(y_true,margin)
fnr=1-tpr
eer=(fpr[np.argmin(abs(fnr-fpr))]+fnr[np.argmin(abs(fnr-fpr))])/2

print(f"ACC={accuracy_score(y_true,y_pred):.4f}")
print(f"EER={eer:.4f}")


# =========================
# 8) SAVE RESULTS CSV
# =========================
out=RESULTS_DIR/"itw_scores_trump.csv"
pd.DataFrame({
    "file":df_trump["file"],
    "label":df_trump["label"],
    "true":y_true,"pred":y_pred,
    "dist":dist,"margin":margin
}).to_csv(out,index=False)

print(f"\nCSV saved → {out}")


# =========================
# 9) SCORE HIST
# =========================
plt.figure(figsize=(7,4))
plt.hist(margin[y_true==0],bins=50,alpha=0.7,label="REAL")
plt.hist(margin[y_true==1],bins=50,alpha=0.7,label="SPOOF")
plt.axvline(0,color="black",lw=2,label="R Boundary")
plt.legend();plt.xlabel("margin = ||z-c||-R")
plt.title("ITW Score Distribution (DeepSVDD)")
plt.savefig(PLOTS_DIR/"itw_hist_scores.png",dpi=300)
plt.close()


# =========================
# 10) UMAP RAW
# =========================
u_raw=umap.UMAP(n_neighbors=30).fit_transform(X_raw)
plt.figure(figsize=(7,6))
plt.scatter(u_raw[:,0],u_raw[:,1],c=y_true,cmap="coolwarm",s=10)
plt.colorbar(label="Spoof=1/Real=0")
plt.title("ITW UMAP (RAW 1024)")
plt.savefig(PLOTS_DIR/"itw_umap_raw.png",dpi=300)
plt.close()


# =========================
# 11) UMAP LATENT
# =========================
u_lat=umap.UMAP(n_neighbors=30).fit_transform(X_lat)
plt.figure(figsize=(7,6))
plt.scatter(u_lat[:,0],u_lat[:,1],c=margin,cmap="viridis",s=10)
plt.colorbar(label="distance vs radius")
plt.title("ITW UMAP (512 latent) — colored by margin")
plt.savefig(PLOTS_DIR/"itw_umap_latent_margin.png",dpi=300)
plt.close()


# =========================
# 12) Sorted margin curve
# =========================
idx=np.argsort(margin)
plt.figure(figsize=(9,4))
plt.plot(margin[idx])
plt.axhline(0,color="black",lw=2,label="R boundary")
plt.legend();plt.ylabel("margin");plt.xlabel("ITW samples sorted")
plt.title("Margin Sorted — How far spoof goes beyond the sphere")
plt.savefig(PLOTS_DIR/"itw_radius_curve.png",dpi=300)
plt.close()


print("\n🔥 DONE — all visualizations generated:")
print(PLOTS_DIR,"\n")
