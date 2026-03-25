import librosa
import numpy as np
import pandas as pd
import os

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
    signal, sample_rate = librosa.load(file_path, duration=3, offset=0.5)
    
    # 1. MFCC — 40 features
    mfcc = librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=40)
    mfcc_mean = np.mean(mfcc, axis=1)
    
    # 2. Chroma — 12 features
    chroma = librosa.feature.chroma_stft(y=signal, sr=sample_rate)
    chroma_mean = np.mean(chroma, axis=1)
    
    # 3. Zero Crossing Rate — 1 feature
    zcr = librosa.feature.zero_crossing_rate(y=signal)
    zcr_mean = np.mean(zcr)
    
    # 4. RMS Energy — 1 feature
    rms = librosa.feature.rms(y=signal)
    rms_mean = np.mean(rms)
    
    # Combine all features
    features = np.concatenate([mfcc_mean, chroma_mean, [zcr_mean], [rms_mean]])
    return features

# Main loop
data = []
data_path = "data/Audio_Speech_Actors_01-24"

print("Extracting features from all audio files...")

for actor in os.listdir(data_path):
    actor_path = os.path.join(data_path, actor)
    for file in os.listdir(actor_path):
        if file.endswith(".wav"):
            emotion_code = file.split("-")[2]
            emotion = emotion_map[emotion_code]
            file_path = os.path.join(actor_path, file)
            features = extract_features(file_path)
            data.append([*features, emotion])

print(f"Done! Total files processed: {len(data)}")

# Save to CSV
columns = (
    [f"mfcc_{i}" for i in range(40)] +
    [f"chroma_{i}" for i in range(12)] +
    ["zcr", "rms", "emotion"]
)
df = pd.DataFrame(data, columns=columns)
df.to_csv("data/features.csv", index=False)
print("Features saved to data/features.csv")
print(f"Feature shape per sample: {df.shape[1]-1} features")