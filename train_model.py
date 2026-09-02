import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib

# Load dataset
df = pd.read_csv("dataset.csv")

X = df.drop(columns=["File"])

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model = IsolationForest(
    contamination=0.1,
    random_state=42
)

model.fit(X_scaled)

# Save model
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Model trained and saved successfully.")