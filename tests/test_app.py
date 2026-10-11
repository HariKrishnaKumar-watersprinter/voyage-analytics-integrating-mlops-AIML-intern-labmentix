"""
test.py — Simple tests for the Flight Price Prediction API
"""

import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime
import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))
from src.flightpriceprediction import app as flight_app
# Mock the database and environment variables BEFORE importing the app
# This prevents pd.read_sql and mlflow from executing at import time
mock_df_data = {
    "from": ["Brasilia (DF)", "Aracaju (SE)", "Sao Paulo (SP)"],
    "to": ["Salvador (BH)", "Natal (RN)", "Aracaju (SE)"],
    "distance": [676.56, 332.1, 392.76],
    "time": [1.76, 0.46, 1.02]
}

@pytest.fixture(autouse=True)
def mock_env_and_db():
    with patch.dict("os.environ", {"MLFLOW_TRACKING_URI": "http://fake:5000"}):
        with patch("pandas.read_sql", return_value=MagicMock()) as mock_read_sql:
            # Simulate the DataFrame behavior for VALID_CITIES initialization
            mock_df = MagicMock()
            mock_df.__getitem__.side_effect = lambda key: mock_df_data[key]
            mock_df.__or__ = lambda other: set(mock_df_data["from"]) | set(mock_df_data["to"])
            mock_read_sql.return_value = mock_df
            
            # Import app here after mocks are in place
            #import app as flight_app
            flight_app.VALID_CITIES = {"Brasilia (DF)", "Aracaju (SE)", "Sao Paulo (SP)"} # Force set valid cities
            yield flight_app

@pytest.fixture
def client(mock_env_and_db):
    """Flask test client fixture."""
    app = mock_env_and_db
    app.app.config["TESTING"] = True
    with app.app.test_client() as client:
        yield client

@pytest.fixture
def app_module(mock_env_and_db):
    """Provides the app module to access its functions and globals."""
    return mock_env_and_db

# ---------------------------------------------------------------------------
# Tests for Validation Logic
# ---------------------------------------------------------------------------
def test_validate_payload_success(app_module):
    payload = {
        "from": "Brasilia (DF)",
        "to": "Salvador (BH)",
        "flightType": "firstClass",
        "agency": "FlyingDrops",
        "date": "10/02/2026"
    }
    errors = app_module.validate_payload(payload)
    assert errors == []

def test_validate_payload_missing_field(app_module):
    payload = {
        "from": "Brasilia (DF)",
        "to": "Salvador (BH)",
        # "flightType" is missing
        "agency": "FlyingDrops",
        "date": "10/02/2026"
    }
    errors = app_module.validate_payload(payload)
    assert len(errors) > 0
    assert "Missing required field: 'flightType'" in errors

def test_validate_payload_invalid_city(app_module):
    payload = {
        "from": "ATL", # Invalid city
        "to": "Salvador (BH)",
        "flightType": "firstClass",
        "agency": "FlyingDrops",
        "date": "10/02/2026"
    }
    errors = app_module.validate_payload(payload)
    assert any("Unknown 'from' city" in e for e in errors)

def test_validate_payload_invalid_date_format(app_module):
    payload = {
        "from": "Brasilia (DF)",
        "to": "Salvador (BH)",
        "flightType": "firstClass",
        "agency": "FlyingDrops",
        "date": "2025-12-25" # Wrong format
    }
    errors = app_module.validate_payload(payload)
    assert any("Invalid 'date' format" in e for e in errors)

def test_validate_payload_same_city(app_module):
    payload = {
        "from": "Brasilia (DF)",
        "to": "Brasilia (DF)",
        "flightType": "economic",
        "agency": "FlyingDrops",
        "date": "12/25/2025"
    }
    errors = app_module.validate_payload(payload)
    assert any("cannot be the same city" in e for e in errors)

# ---------------------------------------------------------------------------
# Tests for API Endpoints
# ---------------------------------------------------------------------------
def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "ok"
    assert data["model"] == "xgboost_flight_prediciton"

def test_predict_endpoint_bad_json(client):
    response = client.post("/predict", data="not json", content_type="text/plain")
    assert response.status_code == 400
    data = response.get_json()
    assert "error" in data

def test_predict_endpoint_validation_error(client):
    payload = {
        "from": "Brasilia (DF)",
        "to": "Salvador (BH)",
        "flightType": "firstClass",
        "agency": "FlyingDrops",
        "date": "10/02/2026"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 400
    data = response.get_json()
    assert data["error"] == "Invalid input"

def test_predict_endpoint_model_not_loaded(client, app_module):
    # Ensure predictor is None
    app_module.PREDICTOR = None
    
    payload = {
        "from": "Brasilia (DF)",
        "to": "Salvador (BH)",
        "flightType": "firstClass",
        "agency": "FlyingDrops",
        "date": "10/02/2026"
    }
    
    # We also need to mock build_feature_row because PREPROCESSOR is None
    with patch.object(app_module, "build_feature_row", side_effect=RuntimeError("Preprocessor model is not loaded in memory.")):
        response = client.post("/predict", json=payload)
        assert response.status_code == 500
        data = response.get_json()
        assert "Model is not loaded" in data["details"] or "Preprocessor" in data["details"]

def test_predict_endpoint_success(client, app_module):
    # Mock the models and feature builder
    mock_predictor = MagicMock()
    mock_predictor.predict.return_value = [1724.49]
    app_module.PREDICTOR = mock_predictor
    
    # Mock build_feature_row to bypass DB lookup and preprocessing
    mock_features = MagicMock()
    with patch.object(app_module, "build_feature_row", return_value=(mock_features, 676.56)):
        payload = {
            "from": "Brasilia (DF)",
            "to": "Salvador (BH)",
            "flightType": "firstClass",
            "agency": "FlyingDrops",
            "date": "10/02/2026"
        }
        response = client.post("/predict", json=payload)
        
        assert response.status_code == 200
        data = response.get_json()
        assert data["predicted_price"] == 1724.49
        assert data["distance"] == 676.56
        assert data["input"] == payload


if __name__ == "__main__":
    pytest.main([__file__])