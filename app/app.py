"""
app/app.py
----------
Flask web application providing:
  • GET  /              → landing page
  • GET  /analysis      → EDA charts dashboard
  • GET  /predict       → prediction form
  • POST /predict       → run models, return results
  • GET  /metrics       → model performance dashboard
  • GET  /api/predict   → JSON REST endpoint
"""

import sys
import os
from pathlib import Path

from flask import send_from_directory

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(
        os.path.join(app.root_path, 'static'),
        'favicon.ico',
        mimetype='image/vnd.microsoft.icon'
    )
  
# Allow imports from project root
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flask import Flask, render_template, request, jsonify, send_from_directory
import joblib
import numpy as np

from utils.helpers import (
    load_encoders, load_age_model, load_gender_model, load_relationship_model,
    build_feature_row, decode_value, read_metrics
)

# ── Flask setup ────────────────────────────────────────────────────────────────
app = Flask(__name__, template_folder="templates", static_folder="static")

# ── Lazy-load models once ──────────────────────────────────────────────────────
_encoders          = None
_age_bundle        = None
_gender_bundle     = None
_rel_bundle        = None


def get_resources():
    global _encoders, _age_bundle, _gender_bundle, _rel_bundle
    if _encoders is None:
        _encoders      = load_encoders()
        _age_bundle    = load_age_model()
        _gender_bundle = load_gender_model()
        _rel_bundle    = load_relationship_model()
    return _encoders, _age_bundle, _gender_bundle, _rel_bundle


# ── Static dropdown options ────────────────────────────────────────────────────
STATES = [
    "Alabama","Alaska","Arizona","Arkansas","California","Colorado","Connecticut",
    "Delaware","Florida","Georgia","Hawaii","Idaho","Illinois","Indiana","Iowa",
    "Kansas","Kentucky","Louisiana","Maine","Maryland","Massachusetts","Michigan",
    "Minnesota","Mississippi","Missouri","Montana","Nebraska","Nevada",
    "New Hampshire","New Jersey","New Mexico","New York","North Carolina",
    "North Dakota","Ohio","Oklahoma","Oregon","Pennsylvania","Rhode Island",
    "South Carolina","South Dakota","Tennessee","Texas","Utah","Vermont",
    "Virginia","Washington","West Virginia","Wisconsin","Wyoming","District of Columbia",
]
MONTHS = [
    "January","February","March","April","May","June",
    "July","August","September","October","November","December",
]
CRIME_TYPES  = ["Murder or Manslaughter", "Manslaughter by Negligence"]
CRIME_SOLVED = ["Yes", "No"]
VICTIM_SEX   = ["Male", "Female", "Unknown"]
WEAPONS = [
    "Blunt Object","Drowning","Drugs","Explosives","Fall","Fire",
    "Firearm","Gun","Handgun","Knife","Poison","Rifle","Shotgun",
    "Strangulation","Suffocation","Unknown",
]
YEARS = list(range(1980, 2015))


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/analysis")
def analysis():
    fig_dir = ROOT / "reports" / "figures"
    charts  = sorted([f.name for f in fig_dir.glob("*.png")]) if fig_dir.exists() else []
    return render_template("analysis.html", charts=charts)


@app.route("/figures/<filename>")
def serve_figure(filename):
    fig_dir = ROOT / "reports" / "figures"
    return send_from_directory(fig_dir, filename)


@app.route("/predict", methods=["GET", "POST"])
def predict():
    result = None
    error  = None

    if request.method == "POST":
        try:
            enc, age_b, gen_b, rel_b = get_resources()

            # ── collect form inputs ──────────────────────────────────────────
            state       = request.form.get("state", "Unknown")
            year        = int(request.form.get("year", 2000))
            month       = request.form.get("month", "January")
            crime_type  = request.form.get("crime_type", "Murder or Manslaughter")
            crime_solved= request.form.get("crime_solved", "Yes")
            victim_sex  = request.form.get("victim_sex", "Unknown")
            victim_age  = int(request.form.get("victim_age", 30))
            victim_count= int(request.form.get("victim_count", 1))
            weapon      = request.form.get("weapon", "Unknown")

            # ── Age prediction ───────────────────────────────────────────────
            age_feat_names = age_b["feature_names"]
            # exclude perpetrator-related columns if they snuck in
            age_row = build_feature_row(
                enc, age_feat_names,
                state, year, month, crime_type, crime_solved,
                victim_sex, victim_age, victim_count, weapon
            )
            pred_age = int(round(age_b["model"].predict(age_row)[0]))

            # ── Gender prediction ────────────────────────────────────────────
            gen_feat_names = gen_b["feature_names"]
            gen_row = build_feature_row(
                enc, gen_feat_names,
                state, year, month, crime_type, crime_solved,
                victim_sex, victim_age, victim_count, weapon
            )
            gen_row_sc    = gen_b["scaler"].transform(gen_row)
            pred_gen_code = int(gen_b["model"].predict(gen_row_sc)[0])
            pred_gender   = decode_value(enc, "Perpetrator Sex", pred_gen_code)

            # ── Relationship prediction ──────────────────────────────────────
            rel_feat_names = rel_b["feature_names"]
            rel_row = build_feature_row(
                enc, rel_feat_names,
                state, year, month, crime_type, crime_solved,
                victim_sex, victim_age, victim_count, weapon
            )
            pred_rel_code = int(rel_b["model"].predict(rel_row)[0])
            pred_rel      = decode_value(enc, "Relationship", pred_rel_code)

            result = {
                "age":          pred_age,
                "gender":       pred_gender,
                "relationship": pred_rel,
            }

        except Exception as exc:
            error = str(exc)

    return render_template(
        "predict.html",
        states=STATES, months=MONTHS, crime_types=CRIME_TYPES,
        crime_solved_opts=CRIME_SOLVED, victim_sex_opts=VICTIM_SEX,
        weapons=WEAPONS, years=YEARS,
        result=result, error=error,
    )


@app.route("/metrics")
def metrics():
    data = read_metrics()
    return render_template("metrics.html", metrics=data)


# ── JSON REST API ──────────────────────────────────────────────────────────────

@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    Accepts JSON:
    {
      "state": "California",
      "year": 2005,
      "month": "June",
      "crime_type": "Murder or Manslaughter",
      "crime_solved": "Yes",
      "victim_sex": "Female",
      "victim_age": 34,
      "victim_count": 1,
      "weapon": "Handgun"
    }
    Returns JSON with age_prediction, gender_prediction, relationship_prediction.
    """
    try:
        data         = request.get_json(force=True)
        state        = data.get("state",        "Unknown")
        year         = int(data.get("year",     2000))
        month        = data.get("month",        "January")
        crime_type   = data.get("crime_type",   "Murder or Manslaughter")
        crime_solved = data.get("crime_solved", "Yes")
        victim_sex   = data.get("victim_sex",   "Unknown")
        victim_age   = int(data.get("victim_age", 30))
        victim_count = int(data.get("victim_count", 1))
        weapon       = data.get("weapon",       "Unknown")

        enc, age_b, gen_b, rel_b = get_resources()

        age_row    = build_feature_row(enc, age_b["feature_names"],
                                       state, year, month, crime_type, crime_solved,
                                       victim_sex, victim_age, victim_count, weapon)
        pred_age   = int(round(age_b["model"].predict(age_row)[0]))

        gen_row    = build_feature_row(enc, gen_b["feature_names"],
                                       state, year, month, crime_type, crime_solved,
                                       victim_sex, victim_age, victim_count, weapon)
        gen_row_sc = gen_b["scaler"].transform(gen_row)
        pred_gender = decode_value(enc, "Perpetrator Sex",
                                   int(gen_b["model"].predict(gen_row_sc)[0]))

        rel_row    = build_feature_row(enc, rel_b["feature_names"],
                                       state, year, month, crime_type, crime_solved,
                                       victim_sex, victim_age, victim_count, weapon)
        pred_rel   = decode_value(enc, "Relationship",
                                  int(rel_b["model"].predict(rel_row)[0]))

        return jsonify({
            "age_prediction":          pred_age,
            "gender_prediction":       pred_gender,
            "relationship_prediction": pred_rel,
        })

    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


# ── entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("[app] Starting Flask server on http://127.0.0.1:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)
