import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score

from xgboost import XGBClassifier


# --------------------------------------------------
# 1. Project folders
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

DATASET_PATH = DATA_DIR / "student_data.csv"


# --------------------------------------------------
# 2. Create sample dataset if it doesn't exist
# --------------------------------------------------

if not DATASET_PATH.exists():

    print("student_data.csv not found.")
    print("Creating sample student dataset...")

    np.random.seed(42)

    n = 1000

    study_hours = np.random.uniform(1, 10, n)
    attendance = np.random.uniform(50, 100, n)
    assignment_marks = np.random.uniform(40, 100, n)
    internal_marks = np.random.uniform(40, 100, n)
    previous_exam_marks = np.random.uniform(40, 100, n)

    previous_grade = []

    for mark in previous_exam_marks:
        if mark >= 85:
            previous_grade.append("A")
        elif mark >= 70:
            previous_grade.append("B")
        elif mark >= 55:
            previous_grade.append("C")
        elif mark >= 40:
            previous_grade.append("D")
        else:
            previous_grade.append("F")

    final_score = (
        study_hours * 4
        + attendance * 0.15
        + assignment_marks * 0.15
        + internal_marks * 0.25
        + previous_exam_marks * 0.30
    )

    final_grade = []

    for score in final_score:
        if score >= 75:
            final_grade.append("A")
        elif score >= 65:
            final_grade.append("B")
        elif score >= 55:
            final_grade.append("C")
        elif score >= 45:
            final_grade.append("D")
        else:
            final_grade.append("F")

    df = pd.DataFrame({
        "study_hours": study_hours,
        "attendance": attendance,
        "assignment_marks": assignment_marks,
        "internal_marks": internal_marks,
        "previous_exam_marks": previous_exam_marks,
        "previous_grade": previous_grade,
        "final_grade": final_grade
    })

    df.to_csv(DATASET_PATH, index=False)

    print("Sample dataset created successfully.")

else:

    print("student_data.csv found.")
    df = pd.read_csv(DATASET_PATH)


# --------------------------------------------------
# 3. Load dataset
# --------------------------------------------------

df = pd.read_csv(DATASET_PATH)

print("\nDataset loaded successfully.")
print("Rows:", len(df))
print("Columns:", list(df.columns))


# --------------------------------------------------
# 4. Define features and target
# --------------------------------------------------

feature_columns = [
    "study_hours",
    "attendance",
    "assignment_marks",
    "internal_marks",
    "previous_exam_marks",
    "previous_grade"
]

target_column = "final_grade"

X = df[feature_columns].copy()
y = df[target_column].copy()


# --------------------------------------------------
# 5. Encode previous grade
# --------------------------------------------------

previous_grade_encoder = LabelEncoder()

X["previous_grade"] = previous_grade_encoder.fit_transform(
    X["previous_grade"]
)


# --------------------------------------------------
# 6. Encode target grade
# --------------------------------------------------

target_encoder = LabelEncoder()

y_encoded = target_encoder.fit_transform(y)


# --------------------------------------------------
# 7. Scale numeric features
# --------------------------------------------------

numeric_columns = [
    "study_hours",
    "attendance",
    "assignment_marks",
    "internal_marks",
    "previous_exam_marks"
]

scaler = StandardScaler()

X[numeric_columns] = scaler.fit_transform(
    X[numeric_columns]
)


# --------------------------------------------------
# 8. Split dataset
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded
)


# --------------------------------------------------
# 9. Train XGBoost model
# --------------------------------------------------

print("\nTraining XGBoost model...")

model = XGBClassifier(
    n_estimators=150,
    max_depth=5,
    learning_rate=0.08,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42,
    eval_metric="mlogloss"
)

model.fit(X_train, y_train)


# --------------------------------------------------
# 10. Test model
# --------------------------------------------------

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nModel training completed.")
print(f"Test Accuracy: {accuracy * 100:.2f}%")


# --------------------------------------------------
# 11. Save model and preprocessing information
# --------------------------------------------------

model_path = MODEL_DIR / "student_grade_model.pkl"
preprocessor_path = MODEL_DIR / "preprocessor.pkl"

joblib.dump(model, model_path)

preprocessor = {
    "feature_columns": feature_columns,
    "numeric_columns": numeric_columns,
    "previous_grade_encoder": previous_grade_encoder,
    "scaler": scaler,
    "target_encoder": target_encoder,
    "accuracy": accuracy
}

joblib.dump(preprocessor, preprocessor_path)


# --------------------------------------------------
# 12. Final message
# --------------------------------------------------

print("\n========================================")
print("DONE!")
print("========================================")
print(f"Model saved to: {model_path}")
print(f"Preprocessor saved to: {preprocessor_path}")
print(f"Accuracy: {accuracy * 100:.2f}%")
print("========================================")