import os
import librosa
import numpy as np
import soundfile as sf
import noisereduce as nr
from scipy.signal import butter, lfilter


RAW_DIR = "weld audio/raw"
OUT_DIR = "weld audio/extracted"
os.makedirs(OUT_DIR, exist_ok=True)


# Parameters 

SR = 48000
FRAME_LEN = 2048
HOP_LEN = 512


ATTENUATION = 0.03       # how silent non-welding parts should be


# Band-pass filter (welding freq)
# Welding arc ~ 2kHz – 10kHz

def bandpass(data, lowcut=2000, highcut=10000, fs=48000, order=4):
    nyq = 0.5 * fs
    b, a = butter(order,
                  [lowcut / nyq, highcut / nyq],
                  btype='band')
    return lfilter(b, a, data)


# Main processing

for file in os.listdir(RAW_DIR):
    if not file.lower().endswith(".wav"):
        continue

    print(f"Processing: {file}")

    audio, sr = librosa.load(
        os.path.join(RAW_DIR, file),
        sr=SR,
        mono=True
    )

    # ---- Band-pass to emphasize welding ----
    filtered = bandpass(audio)

    # ---- RMS Energy ----
    rms = librosa.feature.rms(
        y=filtered,
        frame_length=FRAME_LEN,
        hop_length=HOP_LEN
    )[0]

    RMS_THRESHOLD = max(0.02, 0.25 * np.max(rms))   #Root mean square value

    #print("RMS min:", np.min(rms))
    #print("RMS max:", np.max(rms))

    # ---- Welding mask (frame-level) ----
    weld_mask_frames = rms > RMS_THRESHOLD

    # ---- Expand mask to sample-level ----
    weld_mask = np.zeros_like(audio)

    for i, is_weld in enumerate(weld_mask_frames):
        start = i * HOP_LEN
        end = start + FRAME_LEN
        if end > len(weld_mask):
            end = len(weld_mask)

        if is_weld:
            weld_mask[start:end] = 1.0
        else:
            weld_mask[start:end] = ATTENUATION

    # ---- Noise reduction only on welding ----
    noise_profile = audio[:int(0.5 * sr)]  # initial background noise

    reduced = nr.reduce_noise(
        y=audio,
        y_noise=noise_profile,
        sr=sr,
        prop_decrease=0.85
    )

    # ---- Apply mask (KEEP FULL LENGTH) ----
    final_audio = reduced * weld_mask

    # ---- Normalize safely ----
    final_audio /= np.max(np.abs(final_audio) + 1e-9)

    # ---- Save ----
    out_path = os.path.join(OUT_DIR, f"Fcheck_{file}")
    sf.write(out_path, final_audio, sr)

    print(f" Save successful: {out_path}")

print("\n Sound extraction compeleted ")
