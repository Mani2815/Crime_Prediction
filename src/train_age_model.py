"""
train_age_model.py
------------------
Trains a Random Forest Regressor to predict the perpetrator's age.

Features  : all columns except 'Perpetrator Age'
Target    : Perpetrator Age  (numeric, years)
Metric    : MSE, RMSE, R²
Saved to  : models/age_prediction.pkl
"""

import os
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# ── paths ──────────────────────────────────────────────────────────────────────
BASE_DIR       = Path(__file__).resolve().parents[1]
DATA_PATH      = BASE_DIR / "data" / "processed" / "crime_cleaned.csv"
MODEL_PATH     = BASE_DIR / "models" / "age_prediction.pkl"
REPORTS_DIR    = BASE_DIR / "reports"

TARGET_COL = "Perpetrator Age"


def train_age_model(data_path=DATA_PATH, model_path=MODEL_PATH):
    os.makedirs(BASE_DIR / "models", exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load preprocessed data
    # ------------------------------------------------------------------
    print("[age model] Loading processed data …")
    df = pd.read_csv(data_path)

    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in dataset.")

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    # ------------------------------------------------------------------
    # 2. Train / test split
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"[age model] Train size: {len(X_train):,}  |  Test size: {len(X_test):,}")

    # ------------------------------------------------------------------
    # 3. Model
    # ------------------------------------------------------------------
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=10,
        n_jobs=-1,
        random_state=42,
    )
    print("[age model] Training RandomForestRegressor …")
    model.fit(X_train, y_train)

    # ------------------------------------------------------------------
    # 4. Evaluation
    # ------------------------------------------------------------------
    y_pred = model.predict(X_test)
    mae    = mean_absolute_error(y_test, y_pred)
    mse    = mean_squared_error(y_test, y_pred)
    rmse   = np.sqrt(mse)
    r2     = r2_score(y_test, y_pred)

    metrics = {
        "Mean Absolute Error (MAE)": round(mae, 4),
        "Mean Squared Error (MSE)":  round(mse, 4),
        "Root Mean Squared Error (RMSE)": round(rmse, 4),
        "R² Score": round(r2, 4),
    }
    print("[age model] Evaluation metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    # ------------------------------------------------------------------
    # 5. Save model + metrics
    # ------------------------------------------------------------------
    joblib.dump({"model": model, "feature_names": X.columns.tolist()}, model_path)
    print(f"[age model] Model saved → {model_path}")

    # Save report
    pd.DataFrame([metrics]).to_csv(REPORTS_DIR / "age_model_metrics.csv", index=False)
    print(f"[age model] Metrics saved → {REPORTS_DIR / 'age_model_metrics.csv'}")

    return metrics


if __name__ == "__main__":
    train_age_model()
