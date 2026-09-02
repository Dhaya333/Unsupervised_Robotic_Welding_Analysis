import os
import librosa
import numpy as np
import pandas as pd

# Folder containing cleaned welding sounds
CLEAN_DIR = "Sound_data/processed_sound"

# Audio parameters
SR = 48000
FRAME_LEN = 2048
HOP_LEN = 512

all_features = []

for file in os.listdir(CLEAN_DIR):

    if not file.lower().endswith(".wav"):
        continue

    print("Processing:", file)

    path = os.path.join(CLEAN_DIR, file)

    # Load audio
    audio, sr = librosa.load(path, sr=SR, mono=True)

    # -------- RMS Energy --------
    rms = librosa.feature.rms(
        y=audio,
        frame_length=FRAME_LEN,
        hop_length=HOP_LEN
    )[0]

    rms_mean = np.mean(rms)
    rms_std = np.std(rms)

    # -------- MFCC --------
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=13,
        hop_length=HOP_LEN
    )

    mfcc_means = np.mean(mfcc, axis=1)

    # Combine features
    features = [file, rms_mean, rms_std]
    features.extend(mfcc_means)

    all_features.append(features)

# -------- Dataset columns --------
columns = ["File", "RMS_Mean", "RMS_Std"]

for i in range(13):
    columns.append(f"MFCC_{i+1}")

# Create dataframe
df = pd.DataFrame(all_features, columns=columns)

# Save dataset
df.to_csv("dataset.csv", index=False)

print("\nFeature dataset created successfully.")