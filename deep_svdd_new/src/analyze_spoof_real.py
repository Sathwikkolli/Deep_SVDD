import librosa, librosa.display
import matplotlib.pyplot as plt
import numpy as np

#####################################################################
# 🔥 3 REAL vs SPOOF PAIRS (Already Filled For You)
#####################################################################

pairs = [

    # Pair #1  (The one you already analyzed)
    (
        "/nfs/turbo/umd-hafiz/issf_server_data/famousfigures/Donald_Trump/-/Donald_Trump_00549.wav",
        "/nfs/turbo/umd-hafiz/issf_server_data/famousfigures/Donald_Trump/LLASA/Donald_Trump_00416_LLASA_00001.wav",
    ),

    # Pair #2  (MASKGCT high-similarity sample)
    (
        "/nfs/turbo/umd-hafiz/issf_server_data/famousfigures/Donald_Trump/-/Donald_Trump_01951.wav",
        "/nfs/turbo/umd-hafiz/issf_server_data/famousfigures/Donald_Trump/MASKGCT/Donald_Trump_00108.wav",
    ),

    # Pair #3  (LLASA sample that lies near real cluster)
    (
        "/nfs/turbo/umd-hafiz/issf_server_data/famousfigures/Donald_Trump/-/Donald_Trump_03171.wav",
        "/nfs/turbo/umd-hafiz/issf_server_data/famousfigures/Donald_Trump/LLASA/Donald_Trump_00383_LLASA_00001.wav",
    ),
]

#####################################################################


def load_audio(path):
    x, sr = librosa.load(path, sr=16000)
    return x, sr


def mel_spectrogram(x, sr):
    S = librosa.feature.melspectrogram(
        y=x, sr=sr,
        n_fft=1024, hop_length=256, n_mels=128
    )
    return librosa.power_to_db(S, ref=np.max)


#####################################################################
# 🔥 PROCESS ALL PAIRS
#####################################################################

for i, (real_path, spoof_path) in enumerate(pairs, start=1):

    print(f"\n### Processing Pair {i} ###")
    print("Real  :", real_path)
    print("Spoof :", spoof_path)

    xr, sr = load_audio(real_path)
    xs, sr = load_audio(spoof_path)

    mel_real = mel_spectrogram(xr, sr)
    mel_fake = mel_spectrogram(xs, sr)

    plt.figure(figsize=(14,10))

    # --- Waveforms ---
    plt.subplot(2,2,1)
    plt.title(f"REAL #{i} – Waveform")
    plt.plot(xr, linewidth=0.8)
    plt.ylim(-1,1)

    plt.subplot(2,2,2)
    plt.title(f"SPOOF #{i} – Waveform")
    plt.plot(xs, linewidth=0.8)
    plt.ylim(-1,1)

    # --- Mel Spectrograms ---
    plt.subplot(2,2,3)
    librosa.display.specshow(mel_real, sr=sr, x_axis="time", y_axis="mel", cmap="magma")
    plt.title(f"REAL #{i} – Mel Spectrogram")
    plt.colorbar(format="%+2.0f dB")

    plt.subplot(2,2,4)
    librosa.display.specshow(mel_fake, sr=sr, x_axis="time", y_axis="mel", cmap="magma")
    plt.title(f"SPOOF #{i} – Mel Spectrogram")
    plt.colorbar(format="%+2.0f dB")

    plt.tight_layout()
    out_path = f"compare_pair_{i}.png"
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"✔ Saved → {out_path}")

