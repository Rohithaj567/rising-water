from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, flash, redirect, render_template, request, url_for
from xgboost import XGBClassifier

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
FEATURE_PATH = BASE_DIR / "reports" / "feature_names.json"

with FEATURE_PATH.open(encoding="utf-8") as file:
    FEATURES = json.load(file)

with (MODEL_DIR / "selected_model.json").open(encoding="utf-8") as file:
    MODEL_INFO = json.load(file)

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "change-this-development-key")
app.config["MAX_CONTENT_LENGTH"] = 1 * 1024 * 1024

if MODEL_INFO["format"] == "xgboost_json":
    model = XGBClassifier()
    model.load_model(str(MODEL_DIR / MODEL_INFO["file"]))
else:
    model = joblib.load(MODEL_DIR / MODEL_INFO["file"])

scaler = joblib.load(MODEL_DIR / "standard_scaler.joblib")

FIELD_INFO = {
    "MonsoonIntensity": ("Monsoon intensity", "Strength of monsoon conditions"),
    "TopographyDrainage": ("Topography drainage", "How well the terrain drains water"),
    "RiverManagement": ("River management", "Condition of river management measures"),
    "Deforestation": ("Deforestation", "Extent of forest loss"),
    "Urbanization": ("Urbanization", "Degree of urban development"),
    "ClimateChange": ("Climate change", "Climate-related pressure indicator"),
    "DamsQuality": ("Dam quality", "Condition of dams and reservoirs"),
    "Siltation": ("Siltation", "Accumulation of sediment in waterways"),
    "AgriculturalPractices": ("Agricultural practices", "Impact of local agricultural practices"),
    "Encroachments": ("Encroachments", "Encroachment on natural water channels"),
    "IneffectiveDisasterPreparedness": ("Disaster preparedness", "Effectiveness of disaster preparedness"),
    "DrainageSystems": ("Drainage systems", "Condition of drainage infrastructure"),
    "CoastalVulnerability": ("Coastal vulnerability", "Exposure to coastal flooding"),
    "Landslides": ("Landslides", "Landslide-related vulnerability"),
    "Watersheds": ("Watersheds", "Watershed condition indicator"),
    "DeterioratingInfrastructure": ("Deteriorating infrastructure", "Condition of local infrastructure"),
    "PopulationScore": ("Population score", "Population exposure indicator"),
    "WetlandLoss": ("Wetland loss", "Extent of wetland loss"),
    "InadequatePlanning": ("Inadequate planning", "Land-use and planning risk indicator"),
    "PoliticalFactors": ("Governance factors", "Governance-related risk indicator"),
}

DEFAULTS = {name: 5 for name in FEATURES}


@app.get("/")
def home():
    return render_template("home.html", features=FEATURES, field_info=FIELD_INFO, defaults=DEFAULTS)


@app.post("/predict")
def predict():
    values = {}
    errors = []
    for feature in FEATURES:
        raw_value = request.form.get(feature, "").strip()
        if raw_value == "":
            errors.append(f"Please enter a value for {FIELD_INFO[feature][0]}.")
            continue
        try:
            value = float(raw_value)
        except ValueError:
            errors.append(f"{FIELD_INFO[feature][0]} must be a number.")
            continue
        if not np.isfinite(value) or value < 0 or value > 10:
            errors.append(f"{FIELD_INFO[feature][0]} must be between 0 and 10.")
            continue
        values[feature] = value

    if errors:
        for error in errors:
            flash(error, "error")
        return redirect(url_for("home"))

    input_frame = pd.DataFrame([[values[name] for name in FEATURES]], columns=FEATURES)
    model_input = scaler.transform(input_frame) if MODEL_INFO["name"] == "K-Nearest Neighbors" else input_frame
    probability = float(model.predict_proba(model_input)[0][1])
    risk = "High" if probability >= 0.5 else "Low"
    result = {
        "risk": risk,
        "probability": probability,
        "percentage": round(probability * 100, 1),
        "values": values,
    }
    return render_template("result.html", result=result)


@app.get("/about")
def about():
    return render_template("about.html")


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None, "model": MODEL_INFO["name"], "features": len(FEATURES)}


@app.errorhandler(404)
def not_found(_error):
    return render_template("error.html", code=404, message="The page you requested could not be found."), 404


@app.errorhandler(500)
def server_error(_error):
    return render_template("error.html", code=500, message="Something went wrong. Please try again."), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 5000)), debug=False)
