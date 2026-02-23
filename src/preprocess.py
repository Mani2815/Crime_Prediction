"""
preprocess.py
-------------
Preprocesses the raw Homicide Reports dataset for use in ML model training.
Steps:
  1. Load raw CSV
  2. Handle missing / invalid values
  3. Drop irrelevant columns
  4. Encode categorical features with LabelEncoder
  5. Persist encoders & cleaned CSV to disk
"""

import os
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# ── paths ──────────────────────────────────────────────────────────────────────
BASE_DIR          = Path(__file__).resolve().parents[1]
RAW_DATA_PATH     = BASE_DIR / "data" / "raw"  / "database.csv"
PROCESSED_DIR     = BASE_DIR / "data" / "processed"
PROCESSED_PATH    = PROCESSED_DIR / "crime_cleaned.csv"
ENCODERS_PATH     = BASE_DIR / "models" / "label_encoders.pkl"


def load_data(path: Path) -> pd.DataFrame:
    print(f"[preprocess] Loading data from {path} …")
    df = pd.read_csv(path, low_memory=False)
    print(f"[preprocess] Raw shape: {df.shape}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    • Drop columns that carry no analytical / predictive value.
    • Fix mixed-type Perpetrator Age column (contains 'Unknown' strings).
    • Replace 'Unknown' / blank gender values.
    • Impute missing numerics with column median.
    """
    print("[preprocess] Cleaning data …")

    # --- drop low-value identifier columns -----------------------------------
    drop_cols = [
        "Record ID", "Agency Code", "Agency Name", "Agency Type",
        "City", "Incident", "Record Source",
        "Victim Race", "Victim Ethnicity",
        "Perpetrator Race", "Perpetrator Ethnicity",
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # --- fix Perpetrator Age -------------------------------------------------
    # Raw column contains numeric strings and the literal string "0" for unknown
    df["Perpetrator Age"] = pd.to_numeric(df["Perpetrator Age"], errors="coerce")
    # age == 0 almost always means 'unknown perpetrator'; treat as NaN
    df["Perpetrator Age"] = df["Perpetrator Age"].replace(0, np.nan)
    median_age = df["Perpetrator Age"].median()
    df["Perpetrator Age"] = df["Perpetrator Age"].fillna(median_age)
    df["Perpetrator Age"] = df["Perpetrator Age"].astype(int)

    # --- Victim Age ----------------------------------------------------------
    df["Victim Age"] = pd.to_numeric(df["Victim Age"], errors="coerce")
    df["Victim Age"] = df["Victim Age"].fillna(df["Victim Age"].median()).astype(int)

    # --- categorical unknowns → explicit "Unknown" label --------------------
    cat_cols = [
        "Perpetrator Sex", "Victim Sex", "Weapon",
        "Crime Solved", "Crime Type", "Relationship",
        "Month", "State",
    ]
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")

    # --- numeric fill --------------------------------------------------------
    for col in ["Victim Count", "Perpetrator Count"]:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int)

    print(f"[preprocess] Cleaned shape: {df.shape}")
    return df


def encode_categorical(df: pd.DataFrame):
    """
    LabelEncode every object/string column (except targets handled elsewhere).
    Returns (encoded_df, encoders_dict).
    """
    from sklearn.preprocessing import LabelEncoder

    print("[preprocess] Encoding categorical variables …")
    encoders = {}
    cat_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()

    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
        print(f"  • {col}: {len(le.classes_)} classes")

    return df, encoders


def preprocess(raw_path=RAW_DATA_PATH,
               processed_path=PROCESSED_PATH,
               encoders_path=ENCODERS_PATH) -> pd.DataFrame:

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(BASE_DIR / "models", exist_ok=True)

    df = load_data(raw_path)
    df = clean_data(df)
    df, encoders = encode_categorical(df)

    df.to_csv(processed_path, index=False)
    joblib.dump(encoders, encoders_path)

    print(f"[preprocess] Processed data saved  → {processed_path}")
    print(f"[preprocess] Label encoders saved  → {encoders_path}")
    return df


if __name__ == "__main__":
    preprocess()
