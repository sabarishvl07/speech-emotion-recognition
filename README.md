# Speech Emotion Recognition 🎙️😠😊😢

> A full-stack AI system that detects human emotion from speech audio using signal processing and an ensemble machine learning model — deployed live as a REST API.

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.135-green?logo=fastapi)](https://fastapi.tiangolo.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8-orange?logo=scikit-learn)](https://scikit-learn.org/)
[![Render](https://img.shields.io/badge/Deployed%20on-Render-purple?logo=render)](https://render.com/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

---

## 🌐 Live Demo

**API Base URL:** `https://speech-emotion-recognition-dd7d.onrender.com`

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | API status |
| `/health` | GET | Health check (model loaded status) |
| `/model-info` | GET | Model metadata, features list |
| `/predict` | POST | Upload a `.wav` file → get predicted emotion |
| `/docs` | GET | Interactive Swagger UI |

---

## 🧠 How It Works

```
Audio File (.wav)
      │
      ▼
Feature Extraction (librosa)
  ├─ MFCC (40 coefficients × mean+std)         → 80 features
  ├─ MFCC Delta                                  → 80 features
  ├─ MFCC Delta-Delta                            → 80 features
  ├─ Mel Spectrogram (128 bands × mean+std)     → 256 features
  ├─ Chroma (12 bins × mean+std)                → 24 features
  ├─ Spectral Contrast (7 bands × mean+std)     → 14 features
  ├─ Spectral Rolloff, Bandwidth, ZCR, RMS      → 8 features
  └─ Pitch / F0                                  → 2 features
                                           Total: 544 features
      │
      ▼
StandardScaler (normalization)
      │
      ▼
Voting Ensemble (soft voting)
  ├─ Random Forest (300 trees)
  ├─ Gradient Boosting (200 estimators)
  └─ SVM (RBF kernel, C=100)
      │
      ▼
Predicted Emotion + Confidence Scores
```

---

## 📊 Model Comparison

| Model | Accuracy |
|---|---|
| **Voting Ensemble (RF + GB + SVM)** ✅ | **Best** |
| Random Forest (300 trees) | - |
| Gradient Boosting (200 estimators) | - |
| SVM (RBF, C=100) | - |
| KNN (k=5) | - |

> Run `python src/compare_models.py` to generate the full comparison table with your trained models.

---

## 🎭 Emotions Detected

| Emotion | Label | Icon |
|---|---|---|
| Neutral | `neutral` | 😐 |
| Calm | `calm` | 😌 |
| Happy | `happy` | 😊 |
| Sad | `sad` | 😢 |
| Angry | `angry` | 😠 |
| Fearful | `fearful` | 😨 |
| Disgust | `disgust` | 🤢 |
| Surprised | `surprised` | 😲 |

---

## 📁 Project Structure

```
speech-emotion-recognition/
├── api/
│   └── main.py              # FastAPI backend (REST API)
├── frontend/
│   └── index.html           # Web UI (upload or record audio)
├── src/
│   ├── extract_features.py  # Extracts 544 audio features from RAVDESS dataset
│   ├── train_model.py       # Trains Voting Ensemble (RF + GB + SVM)
│   └── compare_models.py    # Compares multiple ML models
├── notebooks/
│   └── explore.py           # Exploratory data analysis
├── tests/
│   └── test_api.py          # Unit tests (pytest)
├── models/                  # Saved models (.pkl) — generated after training
├── data/                    # RAVDESS dataset + extracted features.csv
├── requirements.txt
├── Procfile                 # Render deployment
├── render.yaml              # Render cloud config
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/sabarishvl07/speech-emotion-recognition.git
cd speech-emotion-recognition
```

### 2. Create & activate virtual environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download RAVDESS Dataset
Download from [Kaggle — RAVDESS Emotional Speech Audio](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio) and extract to `data/Audio_Speech_Actors_01-24/`.

### 5. Extract Features
```bash
python src/extract_features.py
# → Saves data/features.csv (544 features per audio file)
```

### 6. Train the Model
```bash
python src/train_model.py
# → Saves models/ensemble_model.pkl, models/scaler.pkl, models/label_encoder.pkl
```

### 7. Run the API
```bash
# Set FFmpeg path if not in system PATH
set FFMPEG_PATH=C:\path\to\ffmpeg.exe    # Windows
# export FFMPEG_PATH=/usr/bin/ffmpeg     # Linux/Mac

uvicorn api.main:app --reload --port 8000
```

API will be available at: `http://localhost:8000`  
Interactive docs at: `http://localhost:8000/docs`

### 8. Open the Frontend
Simply open `frontend/index.html` in your browser.

---

## 🧪 Running Tests

```bash
pip install pytest httpx soundfile
pytest tests/ -v
```

---

## 🔧 Environment Variables

| Variable | Default | Description |
|---|---|---|
| `FFMPEG_PATH` | `ffmpeg` | Path to FFmpeg binary. Set if not in system PATH. |

---

## 🌍 Real-World Applications

- 📞 **Call Center Monitoring** — detect frustrated customers in real time
- 🧠 **Mental Health Apps** — track emotional patterns over time
- 🚗 **Driver Safety Systems** — alert when driver sounds stressed/fearful
- 👔 **HR Interview Analysis** — emotion-aware candidate assessment
- 📚 **E-Learning** — detect student engagement from voice

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Feature Extraction** | librosa, NumPy |
| **Machine Learning** | scikit-learn (SVM, RF, GBM, VotingClassifier) |
| **Class Imbalance** | imbalanced-learn (SMOTE) |
| **API Backend** | FastAPI, Uvicorn |
| **Audio Conversion** | pydub, FFmpeg |
| **Frontend** | HTML5, Vanilla CSS, Web Audio API |
| **Cloud Deployment** | Render |

---

## 👨‍💻 Author

**Sabarish** — ECE Student  
🔗 [GitHub](https://github.com/sabarishvl07/speech-emotion-recognition)