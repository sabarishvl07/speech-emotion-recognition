from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import librosa
import numpy as np
import joblib
import tempfile
import os
from pydub import AudioSegment

# Works both locally and on Render
ffmpeg_local = r"C:\Users\info\Downloads\ffmpeg-8.1-essentials_build\ffmpeg-8.1-essentials_build\bin\ffmpeg.exe"
if os.path.exists(ffmpeg_local):
    AudioSegment.converter = ffmpeg_local
else:
    AudioSegment.converter = "ffmpeg"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model = joblib.load(os.path.join(BASE_DIR, "models", "svm_model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "models", "scaler.pkl"))

app = FastAPI(title="Speech Emotion Recognition API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def extract_features(file_path):
    signal, sample_rate = librosa.load(file_path, duration=3, offset=0.5)
    mfcc = librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=40)
    mfcc_mean = np.mean(mfcc, axis=1)
    chroma = librosa.feature.chroma_stft(y=signal, sr=sample_rate)
    chroma_mean = np.mean(chroma, axis=1)
    zcr = librosa.feature.zero_crossing_rate(y=signal)
    zcr_mean = np.mean(zcr)
    rms = librosa.feature.rms(y=signal)
    rms_mean = np.mean(rms)
    features = np.concatenate([mfcc_mean, chroma_mean, [zcr_mean], [rms_mean]])
    return features

@app.get("/")
def home():
    return {"message": "Speech Emotion Recognition API is running!"}

@app.post("/predict")
async def predict_emotion(file: UploadFile = File(...)):
    contents = await file.read()
    filename = file.filename or "recording"

    is_webm = "webm" in filename or "webm" in (file.content_type or "")
    suffix = ".webm" if is_webm else ".wav"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    if is_webm:
        wav_path = tmp_path.replace(".webm", "_converted.wav")
        audio = AudioSegment.from_file(tmp_path, format="webm")
        audio.export(wav_path, format="wav")
        os.unlink(tmp_path)
        tmp_path = wav_path

    features = extract_features(tmp_path)
    features_scaled = scaler.transform([features])
    prediction = model.predict(features_scaled)[0]

    probabilities = model.decision_function(features_scaled)[0]
    prob_min = probabilities.min()
    prob_max = probabilities.max()
    normalized = (probabilities - prob_min) / (prob_max - prob_min) * 100

    emotions = model.classes_
    confidence_scores = {
        emotion: round(float(score), 2)
        for emotion, score in zip(emotions, normalized)
    }

    os.unlink(tmp_path)

    return {
        "emotion": prediction,
        "confidence_scores": confidence_scores,
        "message": f"Detected emotion: {prediction}"
    }