"""
Voyage Analytics — Flight Price Prediction API (objective 2)
"""

import json
import os
import time
import threading
from datetime import datetime

import joblib
import pandas as pd
from flask import Flask, jsonify, request
import sys
import pathlib
import mlflow
from mlflow.tracking import MlflowClient
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)

sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))
from database import engine

df = pd.read_sql("SELECT * FROM flight", engine)

# ---------------------------------------------------------------------------
# Global State: Models will be loaded here once at startup
# ---------------------------------------------------------------------------
PREPROCESSOR = None
PREDICTOR = None
TRAINED_DATE = None
CURRENT_VERSIONS = {
    "flight_preprocess_pipeline": None,
    "xgboost_flight_prediciton": None
}

# Configure how often (in seconds) the background thread checks MLflow
POLL_INTERVAL_SECONDS = 86400 

def get_latest_model_uri(model_name: str) -> tuple | None:
    client = MlflowClient()
    try:
        model_versions = client.search_model_versions(f"name='{model_name}'")
        if not model_versions:
            raise ValueError(f"No registered model found with name '{model_name}'")
        
        latest_version = max(model_versions, key=lambda mv: int(mv.version))
        timestamp_ms = latest_version.creation_timestamp
        timestamp_sec = timestamp_ms / 1000.0
        trained_date = datetime.utcfromtimestamp(timestamp_sec)
        model_uri = f"models:/{model_name}/{latest_version.version}"
        print(f"Latest version found for {model_name}: v{latest_version.version}")
        return model_uri, trained_date, latest_version.version
    except Exception as e:
        print(f"Error getting latest model: {e}")
        return None


def load_models_into_state(force_reload: bool = False):
    """Loads both models into global memory. 
    If force_reload=False, it only loads if a new version is detected."""
    global PREPROCESSOR, PREDICTOR, TRAINED_DATE, CURRENT_VERSIONS

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        raise ValueError("Environment variable MLFLOW_TRACKING_URI is not set")

    mlflow.set_tracking_uri(tracking_uri)

    # 1. Load Preprocessor
    prep_result = get_latest_model_uri('flight_preprocess_pipeline')
    if prep_result:
        prep_uri, _, prep_version = prep_result
        # Check if update is needed
        if force_reload or CURRENT_VERSIONS["flight_preprocess_pipeline"] != prep_version:
            print(f"[Auto-Reload] New preprocessor version detected (v{prep_version}). Reloading...")
            try:
                PREPROCESSOR = mlflow.sklearn.load_model(prep_uri)
                CURRENT_VERSIONS["flight_preprocess_pipeline"] = prep_version
                print("[Auto-Reload] Preprocessor loaded successfully!")
            except Exception as e:
                print(f"[Auto-Reload] Error loading preprocessor: {e}")
        elif not force_reload:
            pass # Silently skip if up to date (avoids spamming logs every 60s)

    # 2. Load Predictor
    pred_result = get_latest_model_uri('xgboost_flight_prediciton')
    if pred_result:
        pred_uri, trained_date, pred_version = pred_result
        # Check if update is needed
        if force_reload or CURRENT_VERSIONS["xgboost_flight_prediciton"] != pred_version:
            print(f"[Auto-Reload] New predictor version detected (v{pred_version}). Reloading...")
            try:
                PREDICTOR = mlflow.pyfunc.load_model(pred_uri)
                TRAINED_DATE = trained_date
                CURRENT_VERSIONS["xgboost_flight_prediciton"] = pred_version
                print("[Auto-Reload] Predictor model loaded successfully!")
            except Exception as e:
                print(f"[Auto-Reload] Error loading predictor: {e}")
        elif not force_reload:
            pass


def background_model_poller():
    """Background thread function to periodically check for new models."""
    print(f"[Background Poller] Starting. Will check MLflow every {POLL_INTERVAL_SECONDS} seconds.")
    while True:
        time.sleep(POLL_INTERVAL_SECONDS)
        try:
            # force_reload=False means it only downloads if the version number changed
            load_models_into_state(force_reload=False)
        except Exception as e:
            print(f"[Background Poller] Error during poll: {e}")


def get_underlying_model(pyfunc_model):
    try:
        return pyfunc_model.unwrap_python_model()
    except Exception:
        pass
    try:
        return pyfunc_model._model_impl.python_model
    except Exception:
        pass
    try:
        return pyfunc_model._model_impl
    except Exception:
        pass
    raise AttributeError("Could not extract the underlying Python model")

# Valid categories
VALID_FLIGHT_TYPES = {"economic", "firstClass", "premium"}
VALID_AGENCIES = {"FlyingDrops", "CloudFy", "Rainbow"}
VALID_CITIES = set(df['from']) | set(df['to'])


def build_feature_row(payload: dict) -> pd.DataFrame:
    if PREPROCESSOR is None:
        raise RuntimeError("Preprocessor model is not loaded in memory.")

    from_  = payload["from"]
    to = payload["to"]
    flight_type = payload["flightType"]
    agency = payload["agency"]
    date = payload["date"]

    mask = (df["from"] == from_) & (df["to"] == to)
    filtered_df = df[mask]

    if filtered_df.empty:
        raise ValueError(f"Route '{from_} -> {to}' not found in the DataFrame.")
    
    distance = float(filtered_df["distance"].iloc[0])
    time_val = float(filtered_df["time"].iloc[0])

    date_parsed = datetime.strptime(date, "%m/%d/%Y")
    
    possible_cols = list(PREPROCESSOR.feature_names_in_)
    row = {col: 0 for col in possible_cols}

    row["distance"] = distance
    row["time"] = time_val
    row["day_of_week"] = date_parsed.weekday()
    row["is_weekend"] = int(date_parsed.weekday() in (5, 6))
    row["month_num"] = date_parsed.month
    row["quarter"] = (date_parsed.month - 1) // 3 + 1

    from_col = f"from_{from_}"
    if from_col in row:
        row[from_col] = 1

    to_col = f"to_{to}"
    if to_col in row:
        row[to_col] = 1

    flighttype_col = f"flightType_{flight_type}"
    if flighttype_col in row:
        row[flighttype_col] = 1

    agency_col = f"agency_{agency}"
    if agency_col in row:
        row[agency_col] = 1
        
    df1 = pd.DataFrame([row], columns=possible_cols)
    transformed_data = PREPROCESSOR.transform(df1)
    df1 = pd.DataFrame(transformed_data, columns=possible_cols)
    
    return df1,distance


def validate_payload(payload: dict):
    errors = []
    required = ["from", "to", "flightType", "agency", "date"]

    for field in required:
        if field not in payload or payload[field] in (None, ""):
            errors.append(f"Missing required field: '{field}'")

    if errors:
        return errors

    if payload["from"] not in VALID_CITIES:
        errors.append(f"Unknown 'from' city: '{payload['from']}'. Valid: {VALID_CITIES}")
    if payload["to"] not in VALID_CITIES:
        errors.append(f"Unknown 'to' city: '{payload['to']}'. Valid: {VALID_CITIES}")
    if payload["from"] == payload["to"]:
        errors.append("'from' and 'to' cannot be the same city")
    if payload["flightType"] not in VALID_FLIGHT_TYPES:
        errors.append(f"Invalid 'flightType': '{payload['flightType']}'. Valid: {sorted(VALID_FLIGHT_TYPES)}")
    if payload["agency"] not in VALID_AGENCIES:
        errors.append(f"Invalid 'agency': '{payload['agency']}'. Valid: {sorted(VALID_AGENCIES)}")

    try:
        datetime.strptime(payload["date"], "%m/%d/%Y")
    except (ValueError, TypeError):
        errors.append("Invalid 'date' format. Expected MM/DD/YYYY, e.g. '09/26/2026'")

    return errors


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model": "xgboost_flight_prediciton",
        "current_versions": CURRENT_VERSIONS,
        "trained_on": str(TRAINED_DATE) if TRAINED_DATE else "Unknown",
    })


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True)

    if payload is None:
        return jsonify({"error": "Request body must be valid JSON"}), 400

    errors = validate_payload(payload)
    if errors:
        return jsonify({"error": "Invalid input", "details": errors}), 400

    try:
        X,d = build_feature_row(payload)
        
        if PREDICTOR is None:
            return jsonify({"error": "Prediction failed", "details": "Model is not loaded"}), 500
            
        prediction = PREDICTOR.predict(X)[0]
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": "Prediction failed", "details": str(e)}), 500

    return jsonify({
        "predicted_price": round(float(prediction), 2),
        "input": payload,
        'distance' :d
    }), 200


if __name__ == "__main__":
    print("--- Loading models into memory ---")
    load_models_into_state(force_reload=True)
    
    # Start the background polling thread
    poller_thread = threading.Thread(target=background_model_poller, daemon=True)
    poller_thread.start()
    
    print("--- Starting Flask Server ---")
    app.run(host="0.0.0.0", port=5000, debug=True)
