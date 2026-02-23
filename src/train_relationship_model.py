"""
train_relationship_model.py
---------------------------
Trains an XGBoost Classifier to predict the victim–perpetrator relationship.

Features  : all columns except 'Relationship'
Target    : Relationship  (multi-class)
Metric    : Accuracy, Precision, Recall, F1
Saved to  : models/xgboost_relationship_model.pkl
"""

import os
import pandas as pd
import joblib
from pathlib import Path

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report
)

# ── paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parents[1]
DATA_PATH   = BASE_DIR / "data" / "processed" / "crime_cleaned.csv"
MODEL_PATH  = BASE_DIR / "models" / "xgboost_relationship_model.pkl"
REPORTS_DIR = BASE_DIR / "reports"
ENC_PATH    = BASE_DIR / "models" / "label_encoders.pkl"

TARGET_COL = "Relationship"


def train_relationship_model(data_path=DATA_PATH, model_path=MODEL_PATH):
    os.makedirs(BASE_DIR / "models", exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print("[relationship model] Loading processed data …")
    df   = pd.read_csv(data_path)
    encs = joblib.load(ENC_PATH)

    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found.")

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    n_classes = y.nunique()
    print(f"[relationship model] {n_classes} relationship classes.")

    # ------------------------------------------------------------------
    # 2. Split
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"[relationship model] Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    # ------------------------------------------------------------------
    # 3. Model
    # ------------------------------------------------------------------
    model = xgb.XGBClassifier(
        objective="multi:softmax",
        num_class=n_classes,
        max_depth=6,
        learning_rate=0.1,
        n_estimators=200,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="mlogloss",
        n_jobs=-1,
        random_state=42,
        verbosity=0,
    )
    print("[relationship model] Training XGBoost …")
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    # ------------------------------------------------------------------
    # 4. Evaluation
    # ------------------------------------------------------------------
    y_pred = model.predict(X_test)
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1   = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    metrics = {
        "Accuracy":  round(acc  * 100, 4),
        "Precision": round(prec * 100, 4),
        "Recall":    round(rec  * 100, 4),
        "F1 Score":  round(f1   * 100, 4),
    }
    print("[relationship model] Evaluation metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    le_rel      = encs[TARGET_COL]
    class_names = [str(c) for c in le_rel.classes_]
    present     = sorted(y_test.unique())
    present_names = [class_names[i] for i in present]
    print(classification_report(y_test, y_pred, labels=present, target_names=present_names))

    # ------------------------------------------------------------------
    # 5. Save
    # ------------------------------------------------------------------
    joblib.dump(
        {"model": model, "feature_names": X.columns.tolist(), "n_classes": n_classes},
        model_path
    )
    print(f"[relationship model] Model saved → {model_path}")

    pd.DataFrame([metrics]).to_csv(REPORTS_DIR / "relationship_model_metrics.csv", index=False)
    print(f"[relationship model] Metrics saved → {REPORTS_DIR / 'relationship_model_metrics.csv'}")

    return metrics


if __name__ == "__main__":
    train_relationship_model()
