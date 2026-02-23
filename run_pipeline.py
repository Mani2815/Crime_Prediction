"""
run_pipeline.py
---------------
End-to-end pipeline runner.  Execute this once to:
  1. Preprocess the raw data
  2. Generate EDA charts
  3. Train the age model (Random Forest Regressor)
  4. Train the gender model (MLP Classifier)
  5. Train the relationship model (XGBoost)

After this script completes, launch the web app with:
    python app/app.py
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.preprocess            import preprocess
from src.analysis              import run_analysis
from src.train_age_model       import train_age_model
from src.train_gender_model    import train_gender_model
from src.train_relationship_model import train_relationship_model


BANNER = """
╔══════════════════════════════════════════════════════════╗
║  Crime Data Analysis & Perpetrator Identity Prediction  ║
║                   — Full Pipeline —                      ║
╚══════════════════════════════════════════════════════════╝
"""


def run():
    print(BANNER)
    t0 = time.time()

    # ── Step 1: Preprocess ────────────────────────────────────────────────────
    print("=" * 60)
    print("STEP 1 / 5 — Data Preprocessing")
    print("=" * 60)
    preprocess()

    # ── Step 2: EDA Charts ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 2 / 5 — Exploratory Data Analysis (Chart Generation)")
    print("=" * 60)
    run_analysis()

    # ── Step 3: Age Model ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 3 / 5 — Training Age Prediction Model (Random Forest)")
    print("=" * 60)
    age_metrics = train_age_model()

    # ── Step 4: Gender Model ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 4 / 5 — Training Gender Classification Model (MLP)")
    print("=" * 60)
    gender_metrics = train_gender_model()

    # ── Step 5: Relationship Model ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 5 / 5 — Training Relationship Prediction Model (XGBoost)")
    print("=" * 60)
    rel_metrics = train_relationship_model()

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print("\n" + "=" * 60)
    print(f"✅  Pipeline completed in {elapsed:.1f} seconds")
    print("=" * 60)
    print("\n📊 Final Model Metrics:")
    print(f"  Age model       → R²: {age_metrics.get('R² Score', 'N/A')}")
    print(f"  Gender model    → Accuracy: {gender_metrics.get('Accuracy', 'N/A')}%")
    print(f"  Relation model  → Accuracy: {rel_metrics.get('Accuracy', 'N/A')}%")
    print("\n🚀 To start the web application run:")
    print("     python app/app.py")
    print("   Then open  http://127.0.0.1:5000  in your browser.\n")


if __name__ == "__main__":
    run()
