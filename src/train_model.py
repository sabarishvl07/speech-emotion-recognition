import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load the dataset
print("Loading dataset...")
df = pd.read_csv("data/features.csv")

# Separate features and labels
X = df.drop("emotion", axis=1).values
y = df["emotion"].values

print(f"Dataset shape: {X.shape}")
print(f"Emotions: {np.unique(y)}")

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# Normalize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train SVM model
print("\nTraining SVM model...")
model = SVC(kernel="rbf", C=10, gamma="scale", random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
print("\nDetailed Report:")
print(classification_report(y_test, y_pred))

# Save the model and scaler
joblib.dump(model, "models/svm_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
print("Model and scaler saved to models/ folder!")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Plot confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=np.unique(y),
            yticklabels=np.unique(y))
plt.title("Confusion Matrix")
plt.ylabel("Actual Emotion")
plt.xlabel("Predicted Emotion")
plt.tight_layout()
plt.savefig("models/confusion_matrix.png")
plt.show()
print("Confusion matrix saved to models/confusion_matrix.png")