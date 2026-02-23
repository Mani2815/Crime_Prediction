"""
train_gender_model.py
---------------------
Trains an MLP Classifier to predict perpetrator gender.

Features  : all columns except 'Perpetrator Sex'
Target    : Perpetrator Sex  (binary: Male / Female – 'Unknown' is excluded)
Metric    : Accuracy, Precision, Recall, F1
Saved to  : models/mlp_gender_model.pkl  (contains model + scaler)
"""

import os
import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report
)

# ── paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parents[1]
DATA_PATH   = BASE_DIR / "data" / "processed" / "crime_cleaned.csv"
MODEL_PATH  = BASE_DIR / "models" / "mlp_gender_model.pkl"
REPORTS_DIR = BASE_DIR / "reports"
ENC_PATH    = BASE_DIR / "models" / "label_encoders.pkl"

TARGET_COL = "Perpetrator Sex"


def train_gender_model(data_path=DATA_PATH, model_path=MODEL_PATH):
    os.makedirs(BASE_DIR / "models", exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load data – exclude 'Unknown' gender rows for clean binary task
    # ------------------------------------------------------------------
    print("[gender model] Loading processed data …")
    df   = pd.read_csv(data_path)
    encs = joblib.load(ENC_PATH)

    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found.")

    # Identify numeric code for "Unknown" in the gender encoder
    le_sex = encs[TARGET_COL]
    unknown_code = list(le_sex.classes_).index("Unknown") if "Unknown" in le_sex.classes_ else -1

    if unknown_code >= 0:
        df = df[df[TARGET_COL] != unknown_code].copy()
        print(f"[gender model] Excluded 'Unknown' rows → {len(df):,} rows remain")

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    # ------------------------------------------------------------------
    # 2. Train / test split (stratified)
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"[gender model] Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    # ------------------------------------------------------------------
    # 3. Scale features (MLP is sensitive to feature scale)
    # ------------------------------------------------------------------
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # ------------------------------------------------------------------
    # 4. Model
    # ------------------------------------------------------------------
    model = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        max_iter=200,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42,
        verbose=False,
    )
    print("[gender model] Training MLPClassifier …")
    model.fit(X_train_sc, y_train)

    # ------------------------------------------------------------------
    # 5. Evaluation
    # ------------------------------------------------------------------
    y_pred = model.predict(X_test_sc)
    acc    = accuracy_score(y_test, y_pred)
    prec   = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec    = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1     = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    metrics = {
        "Accuracy":  round(acc  * 100, 4),
        "Precision": round(prec * 100, 4),
        "Recall":    round(rec  * 100, 4),
        "F1 Score":  round(f1   * 100, 4),
    }
    print("[gender model] Evaluation metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print(classification_report(y_test, y_pred, target_names=le_sex.classes_[le_sex.classes_ != "Unknown"]))

    # ------------------------------------------------------------------
    # 6. Save
    # ------------------------------------------------------------------
    joblib.dump(
        {"model": model, "scaler": scaler, "feature_names": X.columns.tolist()},
        model_path
    )
    print(f"[gender model] Model saved → {model_path}")

    pd.DataFrame([metrics]).to_csv(REPORTS_DIR / "gender_model_metrics.csv", index=False)
    print(f"[gender model] Metrics saved → {REPORTS_DIR / 'gender_model_metrics.csv'}")

    return metrics


if __name__ == "__main__":
    train_gender_model()
