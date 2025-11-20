<p align="center">
  <img src="https://img.shields.io/badge/Project-DeepSVDD%20Speaker%20Deepfake%20Detection-blue.svg?style=for-the-badge">
  <img src="https://img.shields.io/badge/Framework-PyTorch-red.svg?style=for-the-badge">
  <img src="https://img.shields.io/badge/Embeddings-WavLM%20Large-green.svg?style=for-the-badge">
  <img src="https://img.shields.io/badge/EER-8.1%25-success.svg?style=for-the-badge">
</p>

<h1 align="center">🔊 DeepSVDD Speaker-Specific Deepfake Detection</h1>

<p align="center">
   <strong>One-Class Deepfake Detection for Donald Trump using WavLM-Large Embeddings + Deep SVDD</strong><br>
   <em>Training exclusively on bonafide audio, evaluating across multiple spoof generator families.</em>
</p>

---

## 🚀 Overview

This repository implements a **speaker-specific deepfake detection pipeline** using:

- **WavLM-Large** pretrained speech embeddings  
- **Deep SVDD** (Deep Support Vector Data Description) following the official paper  
- **One-class learning** — trained only on bonafide (real) audio  
- **Protocol-based evaluation** using `oc_protocol_eval1000.csv`  
- Spoof detection across **10+ generator families** (StyleTTS2, F5TTS, E2TTS, XTTSV2, FishSpeech, MaskGCT, etc.)

This repo is optimized for **Donald Trump**, but the pipeline generalizes to any speaker.

---

## 📦 Repository Structure
```
deep_svdd/
│
├── src/
│   ├── extract_embeddings.py    # WavLM-large embedding extraction
│   ├── train.py                  # DeepSVDD training + evaluation
│   ├── deep_svdd.py              # SVDD model + center c + radius R
│   ├── utils_svdd.py             # Threshold, EER, score calculation
│   ├── utils_data.py             # Embedding loading + DataLoader utils
│   └── utils.py                  # Protocol parsing + file lookup
│
├── embeddings/                   # (ignored) numpy files from extraction
├── models/                       # (ignored) saved DeepSVDD models + results
├── oc_protocol_eval1000.csv      # evaluation protocol
├── .gitignore
└── README.md
```

---

## 🎤 Dataset & Protocol

This project uses the **Famous Figures Dataset (Donald Trump)**:

- `/Bonafide/` → Real audio  
- All other folders → Spoof audio (COZYVOICE2, MaskGCT, XTTSV2, StyleTTS2, etc.)

The protocol file `oc_protocol_eval1000.csv` defines which files belong to:

| Field | Meaning |
|-------|---------|
| `audiofilepath` | original FF_V2 relative path |
| `split` | train / eval |
| `label` | bonafide / spoof (determined by folder) |

A utility script automatically maps filenames → actual HPC paths.

---

## 🎯 Model Pipeline

### **1️⃣ Extract embeddings (WavLM-large)**
```bash
python src/extract_embeddings.py
```

**Outputs:**
```
embeddings/train_real.npy
embeddings/eval_real.npy
embeddings/eval_spoof.npy
```

### **2️⃣ Train DeepSVDD (One-Class / Soft Boundary)**
```bash
python src/train.py
```

This performs:
- Center initialization `c`
- Deep SVDD training (objective: one-class or soft-boundary)
- Distance scoring
- Threshold-based evaluation
- EER calculation
- Saving model + results

**Outputs:**
```
models/deep_svdd_trump.pt
models/results.txt
```

---

## 📈 Performance Summary

### Fixed Threshold Metrics

| Metric | Value |
|--------|-------|
| Threshold | 0.0004636 |
| TPR (Real detection) | 94.89% |
| FPR (Fake as real) | 12.72% |
| Accuracy | 91.9% |

### Equal Error Rate (EER)
```
EER: 0.08107   (≈ 8.1%)
Threshold: 3.72e-04
FPR @ EER: 0.08142
FNR @ EER: 0.08072
```

An **8.1% EER** is very strong for a pure one-class, no-finetuning detector.

---

## 🔧 Installation

### Create Conda Environment
```bash
conda create -n svdd python=3.10
conda activate svdd
```

### Install Dependencies
```bash
pip install numpy torch==2.1.0 torchaudio==2.1.0 transformers==4.38 tqdm scikit-learn
```

---

## ⚙️ Training Configuration

Inside `src/train.py`, you can adjust:
```python
INPUT_DIM = 1024
REP_DIM   = 128
OBJECTIVE = "soft-boundary"   # or "one-class"
NU        = 0.05
LR        = 1e-4
EPOCHS    = 100
BATCH_SIZE = 64
WEIGHT_DECAY = 1e-6
```

---

## 🧪 How to Test a Single Audio File

Support for `test_audio.py` can be added using:

1. Extract WavLM embedding
2. Use `svdd.score(x)`
3. Threshold classify

If you need this functionality, feel free to open an issue or submit a PR.

---

## 🗂 .gitignore
```gitignore
embeddings/
models/
__pycache__/
*.pt
*.npy
*.pyc
.DS_Store
```

---

## 📚 References

- **Deep SVDD**: [Deep One-Class Classification (ICML 2018)](http://proceedings.mlr.press/v80/ruff18a.html)
- **WavLM**: [WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing](https://arxiv.org/abs/2110.13900)
- **Famous Figures Dataset**: [Add link if publicly available]

---

## 🤝 Contributing

PRs are welcome! Feel free to open an issue for:
- Ideas and improvements
- Extensions to new speakers
- Bug fixes or optimizations


---

## ⭐ Support

If this project helped you, please **star the repo ⭐** — it helps visibility and motivates further work!

---

## 📧 Contact

For questions or collaborations, feel free to reach out via GitHub issues.

---

<p align="center">
  Made with ❤️ for advancing deepfake detection research
</p>
