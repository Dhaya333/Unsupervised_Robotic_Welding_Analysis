from flask import Flask
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import numpy as np
import joblib
import librosa
from scipy.signal import butter, lfilter

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Load trained ML model
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

SR = 48000


# -------- Band-pass filter (welding arc freq) --------
def bandpass(data, lowcut=2000, highcut=10000, fs=48000, order=4):
    nyq = 0.5 * fs
    b, a = butter(order,
                  [lowcut / nyq, highcut / nyq],
                  btype='band')
    return lfilter(b, a, data)


# -------- Feature Extraction --------
def extract_features(audio):

    # RMS Energy
    rms = librosa.feature.rms(y=audio)[0]
    rms_mean = np.mean(rms)
    rms_std = np.std(rms)

    # MFCC
    mfcc = librosa.feature.mfcc(y=audio, sr=SR, n_mfcc=13)
    mfcc_means = np.mean(mfcc, axis=1)

    features = [rms_mean, rms_std]
    features.extend(mfcc_means)

    return np.array(features).reshape(1, -1), rms_mean


@socketio.on("audio_chunk")
def handle_audio(data):

    try:
        # Convert incoming audio
        audio = np.frombuffer(data, dtype=np.uint8).astype(np.float32)
        audio = (audio - 128) / 128

        # Apply welding frequency filter
        audio = bandpass(audio)

        # Extract features
        features, rms_mean = extract_features(audio)

        # -------- Ignore surrounding noise --------
        if rms_mean < 0.02:
            emit("prediction", {"status": "NO WELD"})
            return

        # -------- Spectral centroid check --------
        centroid = librosa.feature.spectral_centroid(y=audio, sr=SR)
        centroid_mean = np.mean(centroid)

        if centroid_mean < 1500:
            emit("prediction", {"status": "NO WELD"})
            return

        # -------- Run ML model --------
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)

        result = "OK" if prediction[0] == 1 else "NOT OK"

        emit("prediction", {"status": result})

    except Exception as e:
        emit("prediction", {"status": "ERROR"})


if __name__ == "__main__":
    socketio.run(app, debug=True)