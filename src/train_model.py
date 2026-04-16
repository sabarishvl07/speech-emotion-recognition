import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ─── Logging Setup ───────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# ─── Load Dataset ─────────────────────────────────────────────────────────────
logger.info("Loading dataset...")
df = pd.read_csv("data/features.csv")

X = df.drop("emotion", axis=1).values
y_raw = df["emotion"].values

le = LabelEncoder()
y = le.fit_transform(y_raw)

logger.info(f"Dataset shape : {X.shape}")
logger.info(f"Emotions      : {list(le.classes_)}")
logger.info(f"Class counts  :\n{pd.Series(y_raw).value_counts()}\n")

# ─── Train / Test Split ───────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
logger.info(f"Training samples : {len(X_train)}")
logger.info(f"Testing  samples : {len(X_test)}\n")

# ─── Feature Scaling ──────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# ─── Handle Class Imbalance with SMOTE (fixes neutral underrepresentation) ────
try:
    from imblearn.over_sampling import SMOTE
    logger.info("Applying SMOTE to balance classes...")
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    logger.info(f"After SMOTE - Training samples: {len(X_train)}")
    logger.info(f"New class distribution:\n{pd.Series(le.inverse_transform(y_train)).value_counts()}\n")
except ImportError:
    logger.warning("imbalanced-learn not installed. Running without SMOTE.")
    logger.warning("Install with: pip install imbalanced-learn\n")

# ─── Voting Ensemble (RF + GB + SVM) ─────────────────────────────────────────
logger.info("Building Voting Ensemble (RandomForest + GradientBoosting + SVM)...")

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_split=2,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

gb = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    random_state=42
)

svm = SVC(
    kernel='rbf',
    C=100,
    gamma='scale',
    class_weight='balanced',
    probability=True,     # needed for soft voting
    random_state=42
)

model = VotingClassifier(
    estimators=[('rf', rf), ('gb', gb), ('svm', svm)],
    voting='soft',        # average predicted probabilities
    n_jobs=-1
)

logger.info("Training ensemble (this takes 2-5 minutes)...")
model.fit(X_train, y_train)
logger.info("Training complete!\n")

# ─── Evaluation ───────────────────────────────────────────────────────────────
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
logger.info(f"\n{'='*50}")
logger.info(f"  Model Accuracy : {accuracy * 100:.2f}%")
logger.info(f"{'='*50}\n")
logger.info("Detailed Report:")
logger.info(f"\n{classification_report(y_test, y_pred, target_names=le.classes_)}")

# ─── Save Model, Scaler & Encoder ─────────────────────────────────────────────
joblib.dump(model,  "models/ensemble_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(le,     "models/label_encoder.pkl")
logger.info("Model saved   -> models/ensemble_model.pkl")
logger.info("Scaler saved  -> models/scaler.pkl")
logger.info("Encoder saved -> models/label_encoder.pkl")

# ─── Confusion Matrix ─────────────────────────────────────────────────────────
label_ints = sorted(np.unique(y))
label_names = le.inverse_transform(label_ints)
cm = confusion_matrix(y_test, y_pred, labels=label_ints)

plt.figure(figsize=(12, 9))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=label_names, yticklabels=label_names)
plt.title(f"Confusion Matrix  --  Accuracy: {accuracy*100:.1f}%")
plt.ylabel("Actual Emotion")
plt.xlabel("Predicted Emotion")
plt.tight_layout()
plt.savefig("models/confusion_matrix.png", dpi=150)
plt.show()
logger.info("Confusion matrix saved -> models/confusion_matrix.png")