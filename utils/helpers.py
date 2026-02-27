"""
utils/helpers.py
----------------
Shared helper functions used by the Flask application and training scripts.
"""

import numpy as np
import pandas as pd
import gzip
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"


# ── label-encoder lookup ───────────────────────────────────────────────────────

def load_encoders():
    """Return the dict of LabelEncoders saved by preprocess.py."""
    path = MODELS_DIR / "label_encoders.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Encoders not found at {path}. Run preprocess first.")
    return joblib.load(path)


def encode_value(encoders: dict, col: str, value: str) -> int:
    """
    Safely encode a single string value for *col*.
    If the value was never seen during training, returns the code for 'Unknown'
    (or 0 as a fallback).
    """
    le = encoders.get(col)
    if le is None:
        return 0
    classes = list(le.classes_)
    if value in classes:
        return int(le.transform([value])[0])
    # fallback: use 'Unknown' if present, else first class
    fallback = "Unknown" if "Unknown" in classes else classes[0]
    return int(le.transform([fallback])[0])


def decode_value(encoders: dict, col: str, code: int) -> str:
    """Reverse-map a numeric code back to its original string label."""
    le = encoders.get(col)
    if le is None:
        return str(code)
    try:
        return str(le.inverse_transform([int(code)])[0])
    except Exception:
        return str(code)


# ── model loaders ──────────────────────────────────────────────────────────────

def load_age_model():
    path = MODELS_DIR / "age_prediction.pkl.gz"
    if not path.exists():
        raise FileNotFoundError(f"Age model not found at {path}.")
    with gzip.open(path, "rb") as f:
        return joblib.load(f)


def load_gender_model():
    path = MODELS_DIR / "mlp_gender_model.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Gender model not found at {path}.")
    return joblib.load(path)


def load_relationship_model():
    path = MODELS_DIR / "xgboost_relationship_model.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Relationship model not found at {path}.")
    return joblib.load(path)


# ── feature builder ────────────────────────────────────────────────────────────

def build_feature_row(
    encoders: dict,
    feature_names: list,
    state: str,
    year: int,
    month: str,
    crime_type: str,
    crime_solved: str,
    victim_sex: str,
    victim_age: int,
    victim_count: int,
    weapon: str,
    perp_count: int = 1,
) -> pd.DataFrame:
    """
    Build a single-row DataFrame matching the feature schema used during training.
    Any column not covered by the provided inputs is filled with 0.
    """
    mapping = {
        "State":             encode_value(encoders, "State",       state),
        "Year":              int(year),
        "Month":             encode_value(encoders, "Month",       month),
        "Crime Type":        encode_value(encoders, "Crime Type",  crime_type),
        "Crime Solved":      encode_value(encoders, "Crime Solved",crime_solved),
        "Victim Sex":        encode_value(encoders, "Victim Sex",  victim_sex),
        "Victim Age":        int(victim_age),
        "Victim Count":      int(victim_count),
        "Weapon":            encode_value(encoders, "Weapon",      weapon),
        "Perpetrator Count": int(perp_count),
    }

    row = {feat: mapping.get(feat, 0) for feat in feature_names}
    return pd.DataFrame([row])


# ── model-metrics reader ───────────────────────────────────────────────────────

def read_metrics() -> dict:
    """Read saved CSV metric files and return a unified dict."""
    results = {}
    for name, fname in [
        ("age",          "age_model_metrics.csv"),
        ("gender",       "gender_model_metrics.csv"),
        ("relationship", "relationship_model_metrics.csv"),
    ]:
        path = BASE_DIR / "reports" / fname
        if path.exists():
            results[name] = pd.read_csv(path).iloc[0].to_dict()
    return results
