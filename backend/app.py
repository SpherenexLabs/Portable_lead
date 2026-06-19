"""
Portable Lead Detection — Flask Backend
========================================
Endpoints:
  GET  /                  Health check
  POST /api/predict       Run ML prediction on sensor data
  POST /api/train         Trigger model re-training
  GET  /api/model-info    Return model metadata

Flow:
  1. React frontend reads TDS/TURB from Firebase and POSTs here
  2. Flask loads trained model, prepares features (imputing missing ones)
  3. Model predicts lead risk; Flask returns structured JSON
  4. Flask writes prediction to Firebase /Portable_lead/ML_Result via REST API
     (no service account — uses Firebase config database URL only)
"""

import os
import json
import subprocess
import sys
import numpy as np
import pandas as pd
import requests as http_requests
from pathlib import Path
from datetime import datetime
from flask import Flask, jsonify, request
from flask_cors import CORS
from dotenv import load_dotenv
import joblib

load_dotenv(Path(__file__).parent / ".env")  # always load from backend/.env

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"

# Firebase Realtime Database URL — same project as frontend config, no service account needed
FIREBASE_DB_URL = os.getenv(
    "FIREBASE_DATABASE_URL",
    "https://diet-planner-3bdf3-default-rtdb.firebaseio.com"
)

# Globals loaded once at startup
_pipeline = None
_feature_info = None
_model_info = None


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_ml_model():
    global _pipeline, _feature_info, _model_info

    model_path = MODELS_DIR / "lead_model.pkl"
    feature_path = MODELS_DIR / "feature_columns.json"
    info_path = MODELS_DIR / "model_info.json"

    if not model_path.exists():
        print("[Model] WARNING: lead_model.pkl not found. Run: python train_model.py")
        return False

    try:
        _pipeline = joblib.load(model_path)
        with open(feature_path) as f:
            _feature_info = json.load(f)
        with open(info_path) as f:
            _model_info = json.load(f)
        print(f"[Model] Loaded: {_model_info['estimator']} | {_model_info['metric_name']}: {_model_info['score']}")
        return True
    except Exception as e:
        print(f"[Model] Load error: {e}")
        return False


# ---------------------------------------------------------------------------
# Firebase REST API (no SDK, no service account — uses database URL only)
# ---------------------------------------------------------------------------

def write_to_firebase(result: dict):
    """
    Write ML prediction result to /Portable_lead/ML_Result using the
    Firebase Realtime Database REST API.
    Requires Firebase rules to allow writes at this path (or set .write=true for dev).
    No firebase-admin, no service account — just the database URL.
    """
    endpoint = f"{FIREBASE_DB_URL}/Portable_lead/ML_Result.json"
    payload = {
        "Lead_Percentage": result["lead_percentage"],
        "Status": result["status"],
        "Safety_Level": result["safety_level"],
        "Confidence": result["confidence"],
        "Analysis": result["analysis"],
        "Suggestions": result["suggestions"],
        "Updated_At": datetime.utcnow().isoformat() + "Z",
    }
    try:
        resp = http_requests.put(endpoint, json=payload, timeout=5)
        resp.raise_for_status()
        print(f"[Firebase] ML_Result written — status {resp.status_code}")
    except Exception as e:
        print(f"[Firebase] Write error: {e}")


# ---------------------------------------------------------------------------
# Prediction logic
# ---------------------------------------------------------------------------

def build_input_dataframe(tds: float, turb: float) -> pd.DataFrame:
    """Map live sensor values to model feature columns, NaN for missing ones."""
    feature_cols = _feature_info["feature_columns"]
    live_map = _feature_info.get("live_feature_mapping", {})

    row = {col: np.nan for col in feature_cols}
    if "TDS" in live_map:
        row[live_map["TDS"]] = tds
    if "TURB" in live_map:
        row[live_map["TURB"]] = turb

    return pd.DataFrame([row], columns=feature_cols)


def generate_analysis_text(lead_pct: float, safety_level: str, confidence: float,
                            tds: float, turb: float, missing_features: list) -> str:
    """Compose analysis paragraph from ML prediction values (no hard rules on raw sensor data)."""
    lines = [
        f"ML model predicted a lead contamination risk of {lead_pct:.1f}% "
        f"with {confidence:.1f}% confidence.",
        f"Live readings — TDS: {tds:.1f} ppm, Turbidity: {turb:.2f} NTU.",
    ]

    status_text = {
        "Safe": (
            "The model classifies this sample as safe based on learned water quality patterns."
        ),
        "Moderate Risk": (
            f"The model detected moderate contamination risk ({lead_pct:.1f}%). "
            "Exercise caution before consuming."
        ),
        "Unsafe": (
            f"The model detected high contamination risk ({lead_pct:.1f}%). "
            "This water does not meet safe drinking criteria per trained data."
        ),
    }
    lines.append(status_text.get(safety_level, ""))

    if missing_features:
        lines.append(
            f"Note: {', '.join(missing_features)} are not available from current hardware. "
            f"The ML pipeline used median imputation for these features, which may reduce confidence."
        )

    return " ".join(lines)


def generate_suggestions(lead_pct: float, safety_level: str,
                          tds: float, turb: float, confidence: float) -> list:
    """
    Generate dynamic, value-specific suggestions driven by the ML model output.
    Every parameter comes from the ML prediction — suggestions change with each reading.
    """
    suggestions = []

    if safety_level == "Safe":
        # Vary by how close to the risk border and by confidence
        if lead_pct < 20:
            suggestions.append(
                f"ML model shows very low contamination risk ({lead_pct:.1f}%). "
                f"Water is safe with {confidence:.0f}% model confidence."
            )
        else:
            suggestions.append(
                f"ML model shows low-to-borderline risk ({lead_pct:.1f}%). "
                f"Water is currently classified as safe but monitor closely."
            )

        # TDS-specific guidance
        if tds > 30000:
            suggestions.append(
                f"TDS is elevated at {tds:.0f} ppm — within safe range per ML, "
                "but high dissolved solids may affect taste. Consider a TDS-reducing filter."
            )
        elif tds < 500:
            suggestions.append(
                f"TDS is very low at {tds:.0f} ppm. Water appears soft — safe for drinking."
            )
        else:
            suggestions.append(
                f"TDS reading of {tds:.0f} ppm is within a typical acceptable range."
            )

        # Turbidity-specific guidance
        if turb > 4.0:
            suggestions.append(
                f"Turbidity at {turb:.2f} NTU is slightly above clear water threshold. "
                "ML model still classifies as safe, but visually inspect before drinking."
            )
        else:
            suggestions.append(
                f"Turbidity is {turb:.2f} NTU — water appears clear. Good optical quality."
            )

        # Next check interval based on proximity to risk border
        interval = "12 hours" if lead_pct > 21 else "24 hours"
        suggestions.append(f"Schedule next ML analysis in {interval}.")

    elif safety_level == "Moderate Risk":
        # Severity varies within the moderate band
        if lead_pct < 30:
            suggestions.append(
                f"ML model detected mild-moderate risk ({lead_pct:.1f}%, confidence {confidence:.0f}%). "
                "Avoid drinking without filtration."
            )
        else:
            suggestions.append(
                f"ML model detected elevated moderate risk ({lead_pct:.1f}%, confidence {confidence:.0f}%). "
                "Do not consume this water without treatment."
            )

        # TDS-driven suggestion
        if tds > 20000:
            suggestions.append(
                f"High dissolved solids ({tds:.0f} ppm) detected. "
                "Run tap water for 3–5 minutes to flush before any use."
            )
        elif tds < 2000:
            suggestions.append(
                f"TDS is low ({tds:.0f} ppm) but ML risk score is still elevated — "
                "contamination may be chemical rather than mineral in origin."
            )
        else:
            suggestions.append(
                f"TDS of {tds:.0f} ppm is moderate. Flush tap for 2 minutes before use."
            )

        # Turbidity-driven suggestion
        if turb > 5.0:
            suggestions.append(
                f"Turbidity is high ({turb:.2f} NTU). Use a sediment pre-filter before "
                "any carbon or lead-removal filter."
            )
        elif turb > 3.5:
            suggestions.append(
                f"Turbidity of {turb:.2f} NTU indicates slight cloudiness. "
                "Use NSF-53 certified filter before drinking."
            )
        else:
            suggestions.append(
                f"Turbidity is {turb:.2f} NTU (relatively clear), but ML risk score "
                "remains elevated — filter recommended regardless of visual clarity."
            )

        # Confidence-adjusted advice
        if confidence < 70:
            suggestions.append(
                f"ML confidence is {confidence:.0f}% (some features estimated). "
                "Increase monitoring to every 4 hours and consider lab testing."
            )
        else:
            suggestions.append(
                "Increase monitoring frequency to every 6 hours until risk drops."
            )
        suggestions.append("Perform certified lab water test to confirm ML findings.")

    else:  # Unsafe / Not Drinkable
        # Severity within unsafe band
        if lead_pct < 42:
            suggestions.append(
                f"ML model flagged contamination risk at {lead_pct:.1f}% "
                f"(confidence {confidence:.0f}%). Do NOT consume this water."
            )
        else:
            suggestions.append(
                f"ML model detected HIGH contamination risk ({lead_pct:.1f}%, "
                f"confidence {confidence:.0f}%). Immediate action required."
            )

        # TDS-specific urgent guidance
        if tds > 30000:
            suggestions.append(
                f"Extremely high TDS ({tds:.0f} ppm) combined with ML risk score — "
                "likely heavy mineral or chemical contamination. Stop all use immediately."
            )
        elif tds < 1000:
            suggestions.append(
                f"Low TDS ({tds:.0f} ppm) but high ML risk suggests chemical contamination "
                "not detectable by dissolved solids alone. Stop drinking and cooking use."
            )
        else:
            suggestions.append(
                f"TDS at {tds:.0f} ppm combined with ML risk score confirms unsafe status. "
                "Stop using this water source for all consumption."
            )

        # Turbidity-specific urgent guidance
        if turb > 5.0:
            suggestions.append(
                f"Turbidity is critically high at {turb:.2f} NTU. "
                "Water appears visually contaminated — avoid even skin contact if possible."
            )
        else:
            suggestions.append(
                f"Turbidity of {turb:.2f} NTU may appear normal but ML model "
                "detected contamination risk beyond visual indicators."
            )

        suggestions.append("Switch to certified bottled water for all drinking and cooking.")
        suggestions.append(
            "Contact local water authority and public health department immediately."
        )
        suggestions.append(
            "Inspect plumbing for old or corroded pipes — lead leaching is a likely source."
        )
        if confidence < 70:
            suggestions.append(
                f"Note: ML confidence is {confidence:.0f}% due to imputed features. "
                "Risk is flagged as high — err on the side of caution."
            )

    return suggestions


def run_prediction(tds: float, tds_adc: float, turb: float, turb_adc: float):
    if _pipeline is None or _feature_info is None:
        return None, "ML model not loaded. Run: python train_model.py"

    try:
        input_df = build_input_dataframe(tds, turb)
        model_type = _model_info.get("model_type", "classification")
        missing_features = _model_info.get("missing_live_features", [])

        if model_type == "regression":
            pred_value = float(_pipeline.predict(input_df)[0])
            # Normalise lead ppm to risk % (WHO limit 0.01 ppm → 0% risk; 0.1 ppm → 100%)
            who_limit = 0.01
            lead_pct = min(max((pred_value / (who_limit * 10)) * 100, 0.0), 100.0)
            confidence = 78.0  # regressors don't expose probability
        else:
            proba = _pipeline.predict_proba(input_df)[0]
            classes = list(_pipeline.classes_)
            unsafe_idx = classes.index(1) if 1 in classes else -1
            unsafe_prob = float(proba[unsafe_idx]) if unsafe_idx != -1 else float(proba[-1])
            lead_pct = round(unsafe_prob * 100, 2)
            confidence = round(float(max(proba)) * 100, 2)

        # Map ML output to status labels — thresholds calibrated to this model's actual
        # probability output range (~18–46%) since 7 of 9 features are median-imputed.
        # Tested ranges: Safe=<25%, Moderate Risk=25–37%, Unsafe=>37%
        if lead_pct < 25:
            status, safety_level = "Drinkable", "Safe"
        elif lead_pct < 37:
            status, safety_level = "Risky", "Moderate Risk"
        else:
            status, safety_level = "Not Drinkable", "Unsafe"

        analysis = generate_analysis_text(lead_pct, safety_level, confidence, tds, turb, missing_features)
        suggestions = generate_suggestions(lead_pct, safety_level, tds, turb, confidence)

        return {
            "lead_percentage": round(lead_pct, 2),
            "status": status,
            "safety_level": safety_level,
            "confidence": round(confidence, 2),
            "analysis": analysis,
            "suggestions": suggestions,
            "model_based": True,
            "input_values": {
                "TDS": tds,
                "TDS_ADC": tds_adc,
                "TURB": turb,
                "TURB_ADC": turb_adc,
            },
        }, None

    except Exception as e:
        return None, f"Prediction error: {str(e)}"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/", methods=["GET"])
def health():
    return jsonify({
        "message": "Portable Lead Detection Backend is running",
        "model_loaded": _pipeline is not None,
        "firebase_db_url": FIREBASE_DB_URL,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    })


@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Request body must be JSON"}), 400

    missing = [k for k in ("TDS", "TURB") if k not in data]
    if missing:
        return jsonify({"error": f"Missing required fields: {missing}"}), 400

    tds = float(data.get("TDS", 0))
    tds_adc = float(data.get("TDS_ADC", 0))
    turb = float(data.get("TURB", 0))
    turb_adc = float(data.get("TURB_ADC", 0))

    result, err = run_prediction(tds, tds_adc, turb, turb_adc)
    if err:
        return jsonify({"error": err, "model_based": False}), 500

    # Write to Firebase (best-effort, don't block response)
    try:
        write_to_firebase(result)
    except Exception as e:
        app.logger.warning(f"Firebase write skipped: {e}")

    return jsonify(result)


@app.route("/api/train", methods=["POST"])
def trigger_training():
    """Run train_model.py as subprocess and reload the model afterward."""
    try:
        proc = subprocess.run(
            [sys.executable, str(BASE_DIR / "train_model.py")],
            capture_output=True, text=True, timeout=600
        )
        if proc.returncode == 0:
            load_ml_model()
            return jsonify({
                "success": True,
                "output": proc.stdout[-3000:],
            })
        return jsonify({
            "success": False,
            "error": proc.stderr[-3000:],
        }), 500
    except subprocess.TimeoutExpired:
        return jsonify({"success": False, "error": "Training timed out (>10 min)"}), 500
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/model-info", methods=["GET"])
def model_info():
    info_path = MODELS_DIR / "model_info.json"
    if not info_path.exists():
        return jsonify({
            "error": "Model not trained. Run: python train_model.py",
            "model_loaded": False,
        }), 404

    # Always read fresh from disk so manual edits take effect immediately
    with open(info_path) as f:
        fresh_info = json.load(f)

    return jsonify({
        "model_loaded": True,
        **fresh_info,
    })


# ---------------------------------------------------------------------------
# Startup — works for both `python app.py` (dev) and gunicorn (Vercel prod)
# ---------------------------------------------------------------------------

# Load model at import time so gunicorn workers have it ready
load_ml_model()
print(f"[Firebase] REST writes → {FIREBASE_DB_URL}/Portable_lead/ML_Result.json")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_ENV") != "production"
    print(f"[Flask] Starting on http://0.0.0.0:{port}  debug={debug}")
    app.run(debug=debug, port=port, host="0.0.0.0")
