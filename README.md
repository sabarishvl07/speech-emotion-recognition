# 🎤 Speech Emotion Recognition

A machine learning system that detects human emotions from speech audio using signal processing and Support Vector Machine (SVM) classification.

🔴 [Live Demo](https://speech-emotion-recognition-pearl.vercel.app/) | 📡 [API](https://speech-emotion-recognition-dd7d.onrender.com/docs) | 💻 [GitHub](https://github.com/sabarishvl07/speech-emotion-recognition)

---

## 🎯 What it does
Upload a `.wav` audio file and the system predicts the speaker's emotion from one of 8 categories: **Neutral, Calm, Happy, Sad, Angry, Fearful, Disgust, Surprised**

---

## 🛠️ Tech Stack
| Layer | Technology |
|---|---|
| Signal Processing | LibROSA, NumPy, SciPy |
| Machine Learning | Scikit-learn (SVM) |
| Backend API | FastAPI, Uvicorn |
| Frontend | HTML, CSS, JavaScript |
| Dataset | RAVDESS (1440 audio files) |
| Deployment | Render (backend), Vercel (frontend) |

---

## 📊 Model Performance
- **Accuracy:** 71.18% across 8 emotion classes
- **Features:** 54 features per audio file (MFCC + Chroma + ZCR + RMS Energy)
- **Algorithm:** Support Vector Machine with RBF kernel
- **Dataset:** 1440 audio files from 24 professional actors

---

## 🏗️ Project Architecture
```
User uploads audio
       ↓
Frontend (Vercel)
       ↓
FastAPI Backend (Render)
       ↓
Feature Extraction (MFCC + Chroma + ZCR + RMS)
       ↓
SVM Model Prediction
       ↓
Emotion displayed on UI
```

---

## ⚙️ How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/sabarishvl07/speech-emotion-recognition.git
cd speech-emotion-recognition
```

### 2. Create virtual environment
```bash
python -m venv venv
source venv/Scripts/activate  # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download RAVDESS dataset
Download from [Zenodo](https://zenodo.org/record/1188976) and place in `data/Audio_Speech_Actors_01-24/`

### 5. Extract features and train model
```bash
python src/extract_features.py
python src/train_model.py
```

### 6. Run the API
```bash
uvicorn api.main:app --reload
```

### 7. Open the frontend
Open `frontend/index.html` in your browser

---

## 📁 Project Structure
```
speech-emotion-recognition/
├── api/
│   └── main.py          ← FastAPI backend
├── data/
│   └── features.csv     ← Extracted features
├── frontend/
│   └── index.html       ← Web UI
├── models/
│   ├── svm_model.pkl    ← Trained model
│   └── scaler.pkl       ← Feature scaler
├── src/
│   ├── extract_features.py  ← Feature extraction
│   └── train_model.py       ← Model training
└── requirements.txt
```

---

## 🌍 Real World Applications
- Call center emotion monitoring
- Mental health tracking apps
- Driver safety systems
- HR interview analysis
- E-learning engagement detection

---

## 👨‍💻 Author
**Sabarish** — ECE Student | [GitHub](https://github.com/sabarishvl07)