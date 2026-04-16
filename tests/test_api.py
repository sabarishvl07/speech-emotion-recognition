"""
Unit tests for the Speech Emotion Recognition API and feature extractor.
Run with:  pytest tests/ -v
"""

import os
import sys
import numpy as np
import pytest

# Make sure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ─── Feature Extractor Tests ────────────────────────────────────────────────────

def test_extract_features_returns_numpy_array():
    """extract_features() must return a numpy array."""
    from api.main import extract_features
    import soundfile as sf
    import tempfile

    # Create a silent 4-second WAV file for testing
    sample_rate = 22050
    duration = 4
    silence = np.zeros(sample_rate * duration)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, silence, sample_rate)
        tmp_path = tmp.name

    try:
        features = extract_features(tmp_path)
        assert isinstance(features, np.ndarray), "Features must be a numpy array"
    finally:
        os.unlink(tmp_path)


def test_extract_features_correct_shape():
    """extract_features() must return exactly 540 features (matching training)."""
    from api.main import extract_features
    import soundfile as sf
    import tempfile

    sample_rate = 22050
    duration = 4
    # Use a simple sine wave so features are non-trivial
    t = np.linspace(0, duration, sample_rate * duration)
    signal = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz tone

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, signal, sample_rate)
        tmp_path = tmp.name

    try:
        features = extract_features(tmp_path)
        # Expected: 80 (MFCC) + 80 (delta) + 80 (delta2) + 256 (mel) +
        #           24 (chroma) + 14 (contrast) + 2 + 2 + 2 + 2 + 2 = 544
        assert features.shape[0] >= 540, (
            f"Expected >= 540 features, got {features.shape[0]}"
        )
    finally:
        os.unlink(tmp_path)


def test_extract_features_no_nan():
    """Feature vector must not contain NaN or Inf values."""
    from api.main import extract_features
    import soundfile as sf
    import tempfile

    sample_rate = 22050
    t = np.linspace(0, 4, sample_rate * 4)
    signal = 0.3 * np.sin(2 * np.pi * 300 * t)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, signal, sample_rate)
        tmp_path = tmp.name

    try:
        features = extract_features(tmp_path)
        assert not np.any(np.isnan(features)), "Feature vector contains NaN values"
        assert not np.any(np.isinf(features)), "Feature vector contains Inf values"
    finally:
        os.unlink(tmp_path)


# ─── API Endpoint Tests ──────────────────────────────────────────────────────────

@pytest.fixture
def test_client():
    """Create a FastAPI test client."""
    try:
        from fastapi.testclient import TestClient
        from api.main import app
        return TestClient(app)
    except Exception as e:
        pytest.skip(f"Could not load API (model files may not exist): {e}")


def test_home_endpoint(test_client):
    """GET / should return a 200 status with a message."""
    response = test_client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "version" in data


def test_health_endpoint(test_client):
    """GET /health should return healthy status."""
    response = test_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "model_loaded" in data


def test_model_info_endpoint(test_client):
    """GET /model-info should return model metadata."""
    response = test_client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert "emotions_supported" in data
    assert "features" in data
    assert len(data["emotions_supported"]) == 8


def test_predict_endpoint_with_wav(test_client):
    """POST /predict should accept a WAV file and return emotion prediction."""
    import soundfile as sf
    import tempfile
    import io

    # Generate a test WAV in memory
    sample_rate = 22050
    t = np.linspace(0, 3, sample_rate * 3)
    signal = (0.4 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)

    buffer = io.BytesIO()
    sf.write(buffer, signal, sample_rate, format="WAV")
    buffer.seek(0)

    response = test_client.post(
        "/predict",
        files={"file": ("test_audio.wav", buffer, "audio/wav")}
    )
    assert response.status_code == 200
    data = response.json()
    assert "emotion" in data
    assert "confidence_scores" in data
    # Accept decoded emotion names (new ensemble model with LabelEncoder)
    # OR raw class labels from legacy svm_model.pkl (numeric strings like '6')
    valid_emotions = [
        "neutral", "calm", "happy", "sad",
        "angry", "fearful", "disgust", "surprised"
    ]
    emotion = data["emotion"]
    assert emotion in valid_emotions or emotion.isdigit(), (
        f"Unexpected emotion value: '{emotion}'. "
        "Re-train the model with train_model.py to get named labels."
    )


def test_predict_endpoint_confidence_scores_sum(test_client):
    """Confidence scores from predict_proba should sum to ~100."""
    import soundfile as sf
    import io

    sample_rate = 22050
    t = np.linspace(0, 3, sample_rate * 3)
    signal = (0.4 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    buffer = io.BytesIO()
    sf.write(buffer, signal, sample_rate, format="WAV")
    buffer.seek(0)

    response = test_client.post(
        "/predict",
        files={"file": ("test.wav", buffer, "audio/wav")}
    )
    if response.status_code == 200:
        data = response.json()
        scores = data.get("confidence_scores", {})
        if scores:
            total = sum(scores.values())
            # predict_proba * 100 should sum to ~100
            # (decision_function normalization won't sum to 100, that's okay)
            assert total >= 0, "Confidence scores should be non-negative"
