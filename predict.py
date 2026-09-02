import librosa
import numpy as np
import joblib
import sys

# Load model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

SR = 48000
FRAME_LEN = 2048
HOP_LEN = 512

def extract_features(file_path):

    audio, sr = librosa.load(file_path, sr=SR, mono=True)

    rms = librosa.feature.rms(
        y=audio,
        frame_length=FRAME_LEN,
        hop_length=HOP_LEN
    )[0]

    rms_mean = np.mean(rms)
    rms_std = np.std(rms)

    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    mfcc_means = np.mean(mfcc, axis=1)

    features = [rms_mean, rms_std]
    features.extend(mfcc_means)

    return np.array(features).reshape(1, -1)


# Take audio file from terminal input
test_file = sys.argv[1]

features = extract_features(test_file)

features_scaled = scaler.transform(features)

prediction = model.predict(features_scaled)

if prediction[0] == -1:
    print("RESULT: BAD WELD")
else:
    print("RESULT: GOOD WELD")