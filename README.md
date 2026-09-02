# AI-Driven Unsupervised Robotic Welding Process Analysis Using Sound
unsupervised-welding-analysis

Final year B.Tech project (Artificial Intelligence and Data Science) — St. Joseph College of Engineering, Sriperumbudur, Anna University, May 2026.

**Team:** Udhaya A (212922243058) · Gobinath T (212922243011) · Sanjay T (212922243058)
**Guide:** Mrs. G.S. Jackulin Asha, M.E., HOD Dept. of AI & DS

# Report 

Google drive project link - [Report.pdf](https://drive.google.com/file/d/1dGc8C-_Dp02XK5iDQLSl1m5Y8-LAcC_n/view?usp=drive_link)

---

## 1. Overview

WeldScan is a **non-contact, sound-based weld quality monitoring system** for robotic welding. Instead of inspecting welds visually or with ultrasonic/radiographic testing *after* the process is complete, WeldScan listens to the acoustic signature of the welding arc **while welding is happening** and flags acoustic anomalies in real time.

Because defective weld samples are rare and hard to label in real industrial settings, the system avoids supervised classification entirely. It uses an **unsupervised Isolation Forest** model trained only on *normal* welding sound, and classifies any segment that deviates significantly from that learned baseline as anomalous.

**Core idea:** `Microphone → Signal Preprocessing → Feature Extraction (RMS + MFCC) → StandardScaler → Isolation Forest → Real-time PASS/FAIL verdict on a live web dashboard`

---

## 2. Key Features

- **Real-time acoustic monitoring** via microphone capture and a Flask + WebSocket (Socket.IO) backend
- **Unsupervised anomaly detection** — no labelled defect data required, trained solely on stable/normal weld audio
- **Band-pass filtering (2–10 kHz)** to isolate the welding-arc frequency range from ambient/industrial noise
- **Spectral noise reduction** and **RMS-based segmentation** to keep only active welding regions
- **16-dimensional feature vector** per segment: 3 RMS statistics (mean, std, peak) + 13 MFCC coefficients
- **Live web dashboard** ("WeldScan — Acoustic Quality Monitor") with:
  - Real-time waveform and frequency-spectrum visualizer
  - PASS / FAIL verdict banner (GOOD WELD / BAD WELD / NO WELD)
  - Live diagnostics: peak amplitude, RMS level, dominant frequency, zero-crossing rate
  - Scrolling session log and result history
  - Backend connection status indicator
- **Offline / batch prediction** via a standalone CLI script for testing recorded `.wav` files
- **Low hardware dependency** — runs entirely in software on a standard microphone + PC

---

## 3. System Architecture

```
 Audio            Signal Pre-       Feature          Feature        Anomaly           Real-Time
 Acquisition      Processing        Extraction        Normalization  Detection         Deployment
 ───────────      ─────────────     ────────────      ─────────────  ───────────────   ───────────────
 Welding arc  ──▶  Band-pass    ──▶  RMS features  ──▶  Standard-  ──▶  Isolation   ──▶  Live audio
 (microphone)      filter            (mean/std/peak)     Scaler         Forest           stream (browser)
                   (2–10 kHz)                            (16D →         (ensemble of      │
 Streamed to   ──▶  Noise       ──▶  MFCC (13         zero mean,      isolation          ▼
 system             reduction        coefficients,      unit var.)     trees)          Flask backend
                   (spectral         FFT→Mel→DCT)                                     (preprocess +
                   subtraction)                                                        extract)
                   RMS               16D feature                                          │
                   segmentation ──▶  vector                                                ▼
                   (threshold                                                          WebSocket
                   > 0.02)                                                            (streams result)
                   Amplitude                                                               │
                   normalization                                                           ▼
                                                                                        Web UI
                                                                                    (Normal/Anomaly
                                                                                     verdict + waveform)
```

### Pipeline stages

1. **Audio Acquisition** — Microphone captures continuous welding-arc sound at a fixed sampling rate (48,000 Hz in the training/offline pipeline; 22,050 Hz configuration also used in the standalone deployment variant).
2. **Signal Preprocessing** — Band-pass filter (Butterworth, order 4, 2–10 kHz) removes irrelevant low/high frequency noise; noise reduction (`noisereduce`) suppresses ambient/industrial noise using an initial silence window as the noise profile; RMS-based thresholding masks out idle/non-welding segments; amplitude normalization standardizes signal levels across recordings.
3. **Feature Extraction** — RMS statistics (mean, std, peak) capture arc-stability/energy characteristics; MFCCs (13 coefficients, mean-aggregated) capture the spectral/timbral shape of the sound → concatenated into a 16-D feature vector per segment.
4. **Feature Normalization** — `StandardScaler` transforms features to zero mean / unit variance so no single feature dominates the anomaly score.
5. **Anomaly Detection** — `IsolationForest` (trained with `contamination=0.1`, `random_state=42`, on normal weld audio only) isolates points via random recursive partitioning; points isolated with fewer splits are flagged as anomalies.
6. **Real-Time Deployment** — Flask + Flask-SocketIO backend receives streamed audio chunks over WebSocket from the browser, re-runs preprocessing + feature extraction + scaling + prediction per chunk, and emits a `prediction` event (`OK` / `NOT OK` / `NO WELD`) back to the frontend, which renders the live verdict, waveform, and spectrum.

---

## 4. Repository / File Structure

```
Welding_ML_model/
├── app.py                  # Flask + Flask-SocketIO backend — real-time inference server
├── train_model.py          # Trains the Isolation Forest model + StandardScaler on dataset.csv
├── feature_dataset.py      # Batch feature extraction: cleaned .wav files -> dataset.csv (RMS + MFCC)
├── sound_extract.py        # Raw audio -> cleaned audio: band-pass filter, noise reduction, RMS masking
├── predict.py               # CLI script — offline prediction on a single .wav file
├── dataset.csv              # Extracted feature dataset (RMS + 13 MFCCs per file) used for training
├── model.pkl                # Trained Isolation Forest model (serialized with joblib)
├── scaler.pkl               # Trained StandardScaler (serialized with joblib)
├── index.html               
├── app.js                   # Frontend logic: mic capture, Web Audio API, canvas visualizer, Socket.IO client
├── style.css                  
├── Sound_data/
│   ├── bad_sound/            # (Bad labelled) anomalous weld sound samples, for reference/testing
│   ├── processed_sound/      # Cleaned welding audio used as input to feature_dataset.py
│   └── test/                 # Held-out audio samples for manual testing
├── weld audio/
│   ├── raw/                  # Original unprocessed recordings (input to sound_extract.py)
│   └── extracted/            # Output of sound_extract.py (cleaned/segmented .wav files)
├── .gitignore
├── requirements.txt          # Python dependencies (see separate file)
└── README.md                 
```

> **Note:** `Sound_data/` and `weld audio/` directories are data folders — populate them with your own `.wav` recordings before running the training pipeline; they are not included with source code by default (see `.gitignore`).

---

## 5. How the Pipeline Fits Together (Script Roles)

| Script | Purpose | Input | Output |
|---|---|---|---|
| `sound_extract.py` | Cleans raw welding recordings: band-pass filter → RMS-based welding-region mask → noise reduction → normalize | `weld audio/raw/*.wav` | `weld audio/extracted/Fcheck_*.wav` |
| `feature_dataset.py` | Extracts RMS + MFCC features from cleaned audio and builds the training dataset | `Sound_data/processed_sound/*.wav` | `dataset.csv` |
| `train_model.py` | Scales features and trains the Isolation Forest anomaly detector | `dataset.csv` | `model.pkl`, `scaler.pkl` |
| `predict.py` | Loads the trained model/scaler and classifies a single audio file offline | any `.wav` file (CLI arg) | Console output: `GOOD WELD` / `BAD WELD` |
| `app.py` | Real-time Flask + Socket.IO backend serving live predictions to the browser dashboard | Live audio chunks (WebSocket) | `prediction` events: `OK` / `NOT OK` / `NO WELD` |
| `index.html` / `app.js` / `style.css` | Browser dashboard: captures mic audio, visualizes waveform/spectrum, streams to backend, renders verdict | Microphone (browser) | Live UI: PASS/FAIL banner, metrics, history log |

---

## 6. System Requirements

### 6.1 Hardware
- A standard microphone (USB or built-in) positioned at a safe, consistent distance from the welding arc
- A PC/server capable of running Python 3.9+ and a modern web browser (for live monitoring)
- No specialized/industrial sensing hardware required

### 6.2 Software
- **OS:** Windows 10/11, Linux (Ubuntu 20.04+), or macOS
- **Python:** 3.9 – 3.11 recommended
- **Browser:** Any modern browser with Web Audio API + WebSocket support (Chrome/Edge/Firefox recommended) and microphone permission granted
- **System audio libraries** (Linux):
  - `libsndfile1` (required by `soundfile`/`librosa`)
  - `portaudio19-dev` (required by `sounddevice`, used in the offline/standalone capture variant)

### 6.3 Python Dependencies
See `requirements.txt` for the full pinned list. Core libraries:
- `Flask`, `Flask-Cors`, `Flask-SocketIO`, `python-socketio`, `eventlet` — backend server & real-time WebSocket streaming
- `numpy`, `pandas`, `scipy` — numerical processing and the Butterworth band-pass filter
- `librosa`, `soundfile`, `noisereduce` — audio loading, feature extraction (RMS, MFCC), and noise reduction
- `scikit-learn`, `joblib` — Isolation Forest model, StandardScaler, model serialization
- `sounddevice` — local microphone capture (used by the standalone real-time script variant)

---

## 7. Setup & Usage

### 7.1 Install dependencies
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 7.2 (Optional) Rebuild the model from scratch
Only needed if you're retraining on new welding audio data.

```bash
# 1. Clean raw recordings (band-pass filter + noise reduction + RMS masking)
python sound_extract.py

# 2. Extract RMS + MFCC features into dataset.csv
python feature_dataset.py

# 3. Train the Isolation Forest model + scaler
python train_model.py
```
This produces `model.pkl` and `scaler.pkl`, which are already included in this repo for immediate use.

### 7.3 Run the real-time backend
```bash
python app.py
```
This starts the Flask + Socket.IO server (default: `http://127.0.0.1:5000`).

### 7.4 Launch the dashboard
Open `index.html` in a browser (e.g. via VS Code's Live Server extension on port `5501`, as configured in `.vscode/settings.json`), grant microphone access, select an audio source, and click **Start Analysis**. The dashboard will stream live audio to the backend and display a continuous PASS/FAIL verdict.

### 7.5 Offline prediction on a recorded file
```bash
python predict.py path/to/weld_sample.wav
```
Prints `RESULT: GOOD WELD` or `RESULT: BAD WELD` based on the trained model.

---

## 8. Model Details

- **Algorithm:** Isolation Forest (`sklearn.ensemble.IsolationForest`)
- **Training data:** Normal/stable welding sound only (unsupervised — no defect labels used)
- **Hyperparameters:** `contamination=0.1`, `random_state=42`
- **Feature vector:** 16-D — `[RMS_Mean, RMS_Std, MFCC_1 … MFCC_13]`
- **Normalization:** `StandardScaler` (zero mean, unit variance)
- **Inference logic:** In the live backend, a segment is first checked against RMS-energy and spectral-centroid thresholds to filter out "no weld" silence/noise; only then is it passed through the scaler and Isolation Forest for the `OK` / `NOT OK` verdict.

**Why unsupervised?** Defective welds are rare and inconsistent in real industrial settings, making labelled datasets impractical to build. Isolation Forest instead learns the acoustic baseline of *normal* welding and flags statistically unusual segments as anomalies — avoiding dependency on annotated defect data while remaining computationally efficient enough for real-time use.

---

## 9. Limitations

- Detection accuracy depends on consistent audio quality and microphone placement/distance
- Environmental/industrial noise can distort features if preprocessing is insufficient
- Flags *anomalies*, not specific defect types (no porosity/crack/fusion classification)
- Model may need retraining when moved to a different welding machine/setup
- Subtle defects that don't meaningfully alter the acoustic signature may go undetected

## 10. Future Work

- Multi-sensor fusion (welding current/voltage, thermal imaging, vibration)
- Extending from anomaly detection to defect-type classification (semi-supervised)
- Adaptive/dynamic anomaly-score thresholding for varying noise environments
- Edge deployment on embedded/microcontroller-based AI accelerators
- Deep learning-based feature learning directly from spectrograms (CNN) instead of handcrafted RMS/MFCC features

---

## 11. References

Selected supporting literature (full list in the project report):
- Zhu, Q., Huang, Z., & Li, H. (2026). *Research Progress of Acoustic Monitoring Technology in Welding and Additive Manufacturing Processes.* Micromachines, 17(2).
- Stemmer, G., et al. (2024). *Unsupervised Welding Defect Detection Using Audio and Video.* arXiv:2409.02290.
- Ji, T., et al. (2023). *Deep Learning-Empowered Digital Twin Using Acoustic Signal for Welding Quality Inspection.* Sensors, 23(5).
- Koizumi, Y., et al. (2019). *ToyADMOS: A Dataset of Miniature-Machine Operating Sounds for Anomalous Sound Detection.*
- Tian, B., et al. (2023). *WeldMon: A Cost-effective Ultrasonic Welding Machine Condition Monitoring System.* arXiv:2308.05756.

---

## 12. Authors

- **Udhaya A** — 212922243058
- **Gobinath T** — 212922243011
- **Sanjay T** — 212922243058

*St. Joseph College of Engineering, Sriperumbudur — Anna University, Chennai. May 2026.*
