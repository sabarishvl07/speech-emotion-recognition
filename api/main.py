import os
import logging
import tempfile
import numpy as np
import joblib
import librosa
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydub import AudioSegment

# ─── Logging Setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# ─── FFmpeg Config (local dev + cloud) ─────────────────────────────────────────
ffmpeg_path = os.getenv("FFMPEG_PATH", "ffmpeg")
AudioSegment.converter = ffmpeg_path
logger.info(f"FFmpeg path set to: {ffmpeg_path}")

# ─── Load Models ───────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

try:
    model  = joblib.load(os.path.join(BASE_DIR, "models", "ensemble_model.pkl"))
    scaler = joblib.load(os.path.join(BASE_DIR, "models", "scaler.pkl"))
    le     = joblib.load(os.path.join(BASE_DIR, "models", "label_encoder.pkl"))
    logger.info("✅ Ensemble model, scaler, and label encoder loaded successfully")
except FileNotFoundError:
    # Fallback to SVM model for backward compatibility
    logger.warning("ensemble_model.pkl not found, trying svm_model.pkl as fallback...")
    model  = joblib.load(os.path.join(BASE_DIR, "models", "svm_model.pkl"))
    scaler = joblib.load(os.path.join(BASE_DIR, "models", "scaler.pkl"))
    le     = None
    logger.info("✅ SVM model loaded (fallback)")

EMOTIONS_SUPPORTED = [
    "neutral", "calm", "happy", "sad",
    "angry", "fearful", "disgust", "surprised"
]

# ─── FastAPI App ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Speech Emotion Recognition API",
    description="Detects human emotion from speech audio using an ensemble of SVM + RandomForest + GradientBoosting",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Feature Extraction (matches src/extract_features.py — 540+ features) ─────
def extract_features(file_path: str) -> np.ndarray:
    """
    Extracts 540+ audio features from a WAV file.
    Mirrors the feature set used during model training in src/extract_features.py.
    """
    signal, sample_rate = librosa.load(file_path, duration=4, offset=0.3)
    features = []

    # 1. MFCC — mean + std (40 x 2 = 80 features)
    mfcc = librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=40)
    features.extend(np.mean(mfcc, axis=1))
    features.extend(np.std(mfcc, axis=1))

    # 2. MFCC Delta — rate of change (40 x 2 = 80 features)
    mfcc_delta = librosa.feature.delta(mfcc)
    features.extend(np.mean(mfcc_delta, axis=1))
    features.extend(np.std(mfcc_delta, axis=1))

    # 3. MFCC Delta-Delta — acceleration (40 x 2 = 80 features)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
    features.extend(np.mean(mfcc_delta2, axis=1))
    features.extend(np.std(mfcc_delta2, axis=1))

    # 4. Mel Spectrogram — 128 x 2 = 256 features
    mel = librosa.feature.melspectrogram(y=signal, sr=sample_rate, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    features.extend(np.mean(mel_db, axis=1))
    features.extend(np.std(mel_db, axis=1))

    # 5. Chroma — 12 x 2 = 24 features
    chroma = librosa.feature.chroma_stft(y=signal, sr=sample_rate)
    features.extend(np.mean(chroma, axis=1))
    features.extend(np.std(chroma, axis=1))

    # 6. Spectral Contrast — 7 x 2 = 14 features
    contrast = librosa.feature.spectral_contrast(y=signal, sr=sample_rate)
    features.extend(np.mean(contrast, axis=1))
    features.extend(np.std(contrast, axis=1))

    # 7. Spectral Rolloff — 2 features
    rolloff = librosa.feature.spectral_rolloff(y=signal, sr=sample_rate)
    features.append(np.mean(rolloff))
    features.append(np.std(rolloff))

    # 8. Spectral Bandwidth — 2 features
    bandwidth = librosa.feature.spectral_bandwidth(y=signal, sr=sample_rate)
    features.append(np.mean(bandwidth))
    features.append(np.std(bandwidth))

    # 9. Zero Crossing Rate — 2 features
    zcr = librosa.feature.zero_crossing_rate(y=signal)
    features.append(np.mean(zcr))
    features.append(np.std(zcr))

    # 10. RMS Energy — 2 features
    rms = librosa.feature.rms(y=signal)
    features.append(np.mean(rms))
    features.append(np.std(rms))

    # 11. Pitch (F0) — 2 features
    pitches, _ = librosa.piptrack(y=signal, sr=sample_rate)
    pitch_values = pitches[pitches > 0]
    features.append(np.mean(pitch_values) if len(pitch_values) > 0 else 0.0)
    features.append(np.std(pitch_values) if len(pitch_values) > 0 else 0.0)

    return np.array(features)

# ─── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", tags=["Info"])
def home():
    return {
        "message": "Speech Emotion Recognition API is running!",
        "version": "2.0.0",
        "docs": "/docs"
    }

@app.get("/health", tags=["Info"])
def health_check():
    """Returns the health status of the API and loaded models."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
        "label_encoder_loaded": le is not None,
    }

@app.get("/model-info", tags=["Info"])
def model_info():
    """Returns metadata about the trained model."""
    return {
        "model_type": "VotingClassifier (RandomForest + GradientBoosting + SVM)",
        "dataset": "RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)",
        "emotions_supported": EMOTIONS_SUPPORTED,
        "features": {
            "total": 540,
            "types": [
                "MFCC (mean + std, 40 coefficients)",
                "MFCC Delta (mean + std, 40 coefficients)",
                "MFCC Delta-Delta (mean + std, 40 coefficients)",
                "Mel Spectrogram (mean + std, 128 bands)",
                "Chroma (mean + std, 12 bins)",
                "Spectral Contrast (mean + std, 7 bands)",
                "Spectral Rolloff (mean + std)",
                "Spectral Bandwidth (mean + std)",
                "Zero Crossing Rate (mean + std)",
                "RMS Energy (mean + std)",
                "Pitch / F0 (mean + std)"
            ]
        },
        "class_imbalance": "Handled with SMOTE oversampling",
        "version": "2.0.0"
    }

@app.post("/predict", tags=["Prediction"])
async def predict_emotion(file: UploadFile = File(...)):
    """
    Analyzes an uploaded audio file and returns the detected emotion
    along with confidence scores for all 8 emotion categories.
    """
    logger.info(f"Received file: {file.filename} | content_type: {file.content_type}")

    contents = await file.read()
    filename = file.filename or "recording"

    is_webm = "webm" in filename or "webm" in (file.content_type or "")
    suffix = ".webm" if is_webm else ".wav"

    tmp_path = None
    wav_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        if is_webm:
            wav_path = tmp_path.replace(".webm", "_converted.wav")
            logger.info("Converting WebM → WAV...")
            audio = AudioSegment.from_file(tmp_path, format="webm")
            audio.export(wav_path, format="wav")
            os.unlink(tmp_path)
            tmp_path = None
            process_path = wav_path
        else:
            process_path = tmp_path

        logger.info("Extracting features...")
        features = extract_features(process_path)
        logger.info(f"Feature vector shape: {features.shape}")

        features_scaled = scaler.transform([features])
        prediction_encoded = model.predict(features_scaled)[0]

        # Decode label
        if le is not None:
            emotion = le.inverse_transform([prediction_encoded])[0]
        else:
            emotion = str(prediction_encoded)

        # Confidence scores
        try:
            proba = model.predict_proba(features_scaled)[0]
            if le is not None:
                emotion_labels = le.classes_
            else:
                emotion_labels = model.classes_
            confidence_scores = {
                str(label): round(float(p) * 100, 2)
                for label, p in zip(emotion_labels, proba)
            }
        except AttributeError:
            # Fallback: use decision_function (e.g. for SVM without predict_proba)
            scores = model.decision_function(features_scaled)[0]
            norm_min, norm_max = scores.min(), scores.max()
            normalized = (scores - norm_min) / (norm_max - norm_min + 1e-8) * 100
            emotion_labels = model.classes_
            confidence_scores = {
                str(e): round(float(s), 2)
                for e, s in zip(emotion_labels, normalized)
            }

        logger.info(f"Predicted emotion: {emotion}")

        return {
            "emotion": emotion,
            "confidence_scores": confidence_scores,
            "message": f"Detected emotion: {emotion}",
            "features_extracted": int(features.shape[0])
        }

    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

    finally:
        for p in [tmp_path, wav_path]:
            if p and os.path.exists(p):
                try:
                    os.unlink(p)
                except Exception:
                    pass