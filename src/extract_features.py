import logging
import librosa
import numpy as np
import pandas as pd
import os

# ─── Logging Setup ───────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

emotion_map = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

def extract_features(file_path):
    """
    Extracts 544 audio features from a WAV file using librosa.

    Features extracted:
        - MFCC (40 coefficients, mean + std)             →  80
        - MFCC Delta (mean + std)                         →  80
        - MFCC Delta-Delta (mean + std)                   →  80
        - Mel Spectrogram (128 bands, mean + std)         → 256
        - Chroma (12 bins, mean + std)                    →  24
        - Spectral Contrast (7 bands, mean + std)         →  14
        - Spectral Rolloff (mean + std)                   →   2
        - Spectral Bandwidth (mean + std)                 →   2
        - Zero Crossing Rate (mean + std)                 →   2
        - RMS Energy (mean + std)                         →   2
        - Pitch / F0 (mean + std)                         →   2
                                                   Total: 544

    Args:
        file_path (str): Path to the .wav audio file.

    Returns:
        np.ndarray: 1-D array of 544 float features.
    """
    signal, sample_rate = librosa.load(file_path, duration=4, offset=0.3)

    features = []

    # 1. MFCC — mean + std (40 x 2 = 80 features)
    mfcc = librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=40)
    features.extend(np.mean(mfcc, axis=1))
    features.extend(np.std(mfcc, axis=1))

    # 2. MFCC Delta — captures rate of change (40 x 2 = 80 features)
    mfcc_delta = librosa.feature.delta(mfcc)
    features.extend(np.mean(mfcc_delta, axis=1))
    features.extend(np.std(mfcc_delta, axis=1))

    # 3. MFCC Delta-Delta — acceleration of change (40 x 2 = 80 features)
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
    pitches, magnitudes = librosa.piptrack(y=signal, sr=sample_rate)
    pitch_values = pitches[pitches > 0]
    features.append(np.mean(pitch_values) if len(pitch_values) > 0 else 0.0)
    features.append(np.std(pitch_values) if len(pitch_values) > 0 else 0.0)

    return np.array(features)


# ─── Main Loop ────────────────────────────────────────────────────────────────
data = []
data_path = "data/Audio_Speech_Actors_01-24"

logger.info("Extracting features from all audio files...")
failed = 0

for actor in os.listdir(data_path):
    actor_path = os.path.join(data_path, actor)
    if not os.path.isdir(actor_path):
        continue
    for file in os.listdir(actor_path):
        if file.endswith(".wav"):
            try:
                emotion_code = file.split("-")[2]
                emotion = emotion_map[emotion_code]
                file_path = os.path.join(actor_path, file)
                features = extract_features(file_path)
                data.append([*features, emotion])
            except Exception as e:
                failed += 1
                logger.warning(f"Skipped {file}: {e}")

logger.info(f"Done! Total files processed: {len(data)} | Skipped: {failed}")

# Build column names dynamically
n_features = len(data[0]) - 1
columns = [f"feat_{i}" for i in range(n_features)] + ["emotion"]

df = pd.DataFrame(data, columns=columns)
df.to_csv("data/features.csv", index=False)
logger.info(f"Features saved to data/features.csv  ({n_features} features per sample)")