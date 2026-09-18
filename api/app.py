"""
Voyage Analytics — Flight Price Prediction API (objective 2)

Loads the trained Random Forest model + metadata produced by the notebook
(Section 9) and serves real-time price predictions.

Run locally:
    python app.py
Then POST to http://localhost:5000/predict
"""

import json
import os
from datetime import datetime

import joblib
import pandas as pd
from flask import Flask, jsonify, request

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Load model + metadata once at startup (not per-request — this is expensive)
# ---------------------------------------------------------------------------
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

model = joblib.load(os.path.join(MODEL_DIR, "flight_price_model.pkl"))

with open(os.path.join(MODEL_DIR, "model_metadata.json")) as f:
    metadata = json.load(f)

with open(os.path.join(MODEL_DIR, "route_lookup.json")) as f:
    route_lookup = json.load(f)

FEATURE_COLUMNS = metadata["feature_columns"]

# Valid categories — used for input validation, derived from what the model
# actually saw during training (anything else would be an unseen category).
VALID_FLIGHT_TYPES = {"economic", "firstClass", "premium"}
VALID_AGENCIES = {"FlyingDrops", "CloudFy", "Rainbow"}
VALID_CITIES = sorted({r.split("|")[0] for r in route_lookup} | {r.split("|")[1] for r in route_lookup})


def build_feature_row(payload: dict) -> pd.DataFrame:
    """
    Turns a raw API request into a single-row DataFrame with exactly the
    columns (name + order) the model was trained on — this MUST mirror the
    encoding logic in the notebook's Section 6, or predictions will be silently
    wrong instead of erroring out.
    """
    from_city = payload["from"]
    to_city = payload["to"]
    flight_type = payload["flightType"]
    agency = payload["agency"]
    date_str = payload["date"]  # expected format: MM/DD/YYYY, matches training data

    route_key = f"{from_city}|{to_city}"
    if route_key not in route_lookup:
        raise ValueError(
            f"Unknown route '{from_city} -> {to_city}'. "
            f"No distance/time on record for this from/to pair."
        )
    distance = route_lookup[route_key]["distance"]
    time_val = route_lookup[route_key]["time"]

    date_parsed = datetime.strptime(date_str, "%m/%d/%Y")

    row = {col: 0 for col in FEATURE_COLUMNS}

    row["distance"] = distance
    row["time"] = time_val
    row["day_of_week"] = date_parsed.weekday()
    row["is_weekend"] = int(date_parsed.weekday() in (5, 6))
    row["month_num"] = date_parsed.month
    row["quarter"] = (date_parsed.month - 1) // 3 + 1

    # one-hot columns — only set the ones that exist (drop_first=True dropped
    # one baseline category per original column during training)
    from_col = f"from_{from_city}"
    if from_col in row:
        row[from_col] = 1

    to_col = f"to_{to_city}"
    if to_col in row:
        row[to_col] = 1

    flighttype_col = f"flightType_{flight_type}"
    if flighttype_col in row:
        row[flighttype_col] = 1
    # if flight_type == 'economic', both flightType_firstClass and
    # flightType_premium correctly stay 0 — that IS how the model encodes it.

    agency_col = f"agency_{agency}"
    if agency_col in row:
        row[agency_col] = 1

    return pd.DataFrame([row], columns=FEATURE_COLUMNS)


def validate_payload(payload: dict):
    """Returns a list of error strings; empty list means valid."""
    errors = []
    required = ["from", "to", "flightType", "agency", "date"]

    for field in required:
        if field not in payload or payload[field] in (None, ""):
            errors.append(f"Missing required field: '{field}'")

    if errors:
        return errors  # don't bother checking values if fields are missing

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
    """Liveness check — also used by Kubernetes later (objective 4)."""
    return jsonify({
        "status": "ok",
        "model": metadata["model_name"],
        "trained_on": metadata["trained_on"],
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
        X = build_feature_row(payload)
        prediction = model.predict(X)[0]
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        # anything unexpected — don't leak internals, but don't hide it either
        return jsonify({"error": "Prediction failed", "details": str(e)}), 500

    return jsonify({
        "predicted_price": round(float(prediction), 2),
        "input": payload,
        "model": metadata["model_name"],
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)