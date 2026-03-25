from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import librosa
import numpy as np
import joblib
import tempfile
import os

# Load the trained model and scaler
model = joblib.load("models/svm_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# Initialize FastAPI app
app = FastAPI(title="Speech Emotion Recognition API")

# Allow frontend to talk to backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def extract_features(file_path):
    signal, sample_rate = librosa.load(file_path, duration=3, offset=0.5)
    
    # MFCC
    mfcc = librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=40)
    mfcc_mean = np.mean(mfcc, axis=1)
    
    # Chroma
    chroma = librosa.feature.chroma_stft(y=signal, sr=sample_rate)
    chroma_mean = np.mean(chroma, axis=1)
    
    # ZCR
    zcr = librosa.feature.zero_crossing_rate(y=signal)
    zcr_mean = np.mean(zcr)
    
    # RMS Energy
    rms = librosa.feature.rms(y=signal)
    rms_mean = np.mean(rms)
    
    features = np.concatenate([mfcc_mean, chroma_mean, [zcr_mean], [rms_mean]])
    return features

@app.get("/")
def home():
    return {"message": "Speech Emotion Recognition API is running!"}

@app.post("/predict")
async def predict_emotion(file: UploadFile = File(...)):
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        contents = await file.read()
        tmp.write(contents)
        tmp_path = tmp.name
    
    # Extract features and predict
    features = extract_features(tmp_path)
    features_scaled = scaler.transform([features])
    prediction = model.predict(features_scaled)[0]
    
    # Clean up temp file
    os.unlink(tmp_path)
    
    return {
        "emotion": prediction,
        "message": f"Detected emotion: {prediction}"
    }