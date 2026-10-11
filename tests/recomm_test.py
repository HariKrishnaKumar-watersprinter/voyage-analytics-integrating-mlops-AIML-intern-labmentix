import os
import pytest
from unittest.mock import patch, MagicMock

# Import the functions to test from your streamlit script
# Note: If streamlit is not installed in the testing environment, 
# you might need to mock 'streamlit' before importing, or use pytest-streamlit.
from test import get_latest_model_uri, load_model_for_inference, get_underlying_model

# ==========================================
# 1. Tests for MLflow Model Retrieval
# ==========================================

@patch("test.MlflowClient")
def test_get_latest_model_uri_success(mock_client_cls):
    """Test successful retrieval of the latest model URI."""
    # Arrange
    mock_mv1 = MagicMock()
    mock_mv1.version = "1"
    mock_mv2 = MagicMock()
    mock_mv2.version = "2"
    
    mock_client_instance = mock_client_cls.return_value
    mock_client_instance.search_model_versions.return_value = [mock_mv1, mock_mv2]
    
    # Act
    uri = get_latest_model_uri("my_model")
    
    # Assert
    assert uri == "models:/my_model/2"
    mock_client_instance.search_model_versions.assert_called_once_with("name='my_model'")

@patch("test.MlflowClient")
def test_get_latest_model_uri_no_models(mock_client_cls):
    """Test behavior when no models are registered."""
    mock_client_instance = mock_client_cls.return_value
    mock_client_instance.search_model_versions.return_value = []
    
    uri = get_latest_model_uri("missing_model")
    assert uri is None

# ==========================================
# 2. Tests for Model Loading
# ==========================================

@patch("test.get_latest_model_uri")
@patch("test.mlflow")
def test_load_model_for_inference_success(mock_mlflow, mock_get_uri):
    """Test successful model loading."""
    mock_get_uri.return_value = "models:/cosine_similarity_model/1"
    mock_loaded_model = MagicMock()
    mock_mlflow.pyfunc.load_model.return_value = mock_loaded_model
    
    with patch.dict(os.environ, {"MLFLOW_TRACKING_URI": "http://fake-tracking-uri"}):
        model = load_model_for_inference()
        
    assert model == mock_loaded_model
    mock_mlflow.set_tracking_uri.assert_called_once_with("http://fake-tracking-uri")
    mock_mlflow.pyfunc.load_model.assert_called_once_with("models:/cosine_similarity_model/1")

def test_load_model_for_inference_missing_env():
    """Test that loading fails if MLFLOW_TRACKING_URI is not set."""
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError, match="Environment variable MLFLOW_TRACKING_URI is not set"):
            load_model_for_inference()

@patch("test.get_latest_model_uri", return_value=None)
@patch("test.mlflow")
def test_load_model_for_inference_no_registered_model(mock_mlflow, mock_get_uri):
    """Test that loading returns None if no model URI is found."""
    with patch.dict(os.environ, {"MLFLOW_TRACKING_URI": "http://fake-tracking-uri"}):
        model = load_model_for_inference()
        
    assert model is None

# ==========================================
# 3. Tests for Model Unwrapping
# ==========================================

def test_get_underlying_model_method_1():
    """Test unwrapping using Method 1 (unwrap_python_model)."""
    mock_pyfunc = MagicMock()
    mock_real_model = MagicMock()
    mock_pyfunc.unwrap_python_model.return_value = mock_real_model
    
    result = get_underlying_model(mock_pyfunc)
    assert result == mock_real_model

def test_get_underlying_model_method_2():
    """Test unwrapping falling back to Method 2 (_model_impl.python_model)."""
    mock_pyfunc = MagicMock()
    mock_real_model = MagicMock()
    
    # Method 1 fails
    mock_pyfunc.unwrap_python_model.side_effect = AttributeError("Failed")
    # Method 2 succeeds
    mock_pyfunc._model_impl.python_model = mock_real_model
    
    result = get_underlying_model(mock_pyfunc)
    assert result == mock_real_model

def test_get_underlying_model_method_3():
    """Test unwrapping falling back to Method 3 (_model_impl)."""
    mock_pyfunc = MagicMock()
    mock_real_model = MagicMock()
    
    # Method 1 & 2 fail
    mock_pyfunc.unwrap_python_model.side_effect = AttributeError("Failed")
    type(mock_pyfunc._model_impl).python_model = property(lambda self: (_ for _ in ()).throw(AttributeError("Failed")))
    
    # Method 3 succeeds
    mock_pyfunc._model_impl = mock_real_model
    
    result = get_underlying_model(mock_pyfunc)
    assert result == mock_real_model

def test_get_underlying_model_all_fail():
    """Test that an AttributeError is raised if all unwrapping methods fail."""
    mock_pyfunc = MagicMock()
    
    mock_pyfunc.unwrap_python_model.side_effect = AttributeError("Failed")
    type(mock_pyfunc._model_impl).python_model = property(lambda self: (_ for _ in ()).throw(AttributeError("Failed")))
    type(mock_pyfunc)._model_impl = property(lambda self: (_ for _ in ()).throw(AttributeError("Failed")))
    
    with pytest.raises(AttributeError, match="Could not extract the underlying Python model"):
        get_underlying_model(mock_pyfunc)

# ==========================================
# 4. Tests for Streamlit UI Logic
# ==========================================

@patch("test.travel_rec")
@patch("test.st")
def test_streamlit_ui_renders(mock_st, mock_travel_rec_cls):
    """Test that the Streamlit UI components render correctly with mock data."""
    # Import the streamlit function here to apply mocks cleanly
    from test import stramlit_trav_recomm
    
    # Mock the unwrapped model in session state
    mock_rec_model = MagicMock()
    mock_rec_model.user_places.index = [1, 2, 3] # Mock user indexes
    mock_st.session_state = {"unwrapped_rec_model": mock_rec_model}
    
    # Mock the travel_rec hotel data
    mock_hotel_instance = MagicMock()
    mock_hotel_instance.__getitem__.side_effect = lambda key: {
        "place": MagicMock(unique=MagicMock(return_value=["Paris", "Tokyo"])),
        "price": MagicMock(max=MagicMock(return_value=500))
    }[key]
    mock_travel_rec_cls.return_value = mock_hotel_instance
    
    # Mock Streamlit widget return values
    mock_st.columns.return_value = (MagicMock(), MagicMock())
    mock_st.slider.side_effect = [300.0, 5] # budget, topn
    mock_st.selectbox.side_effect = ["Paris", "1"] # place, user
    
    # Mock the recommendation output
    import pandas as pd
    mock_df = pd.DataFrame({
        "name": ["Hotel A"], "place": ["Paris"], "bookings": [10], 
        "avg_days": [2], "avg_price_per_day": [100], "avg_total": [200], 
        "score": [0.9], "visited_before": [False]
    })
    mock_rec_model.recommend.return_value = mock_df
    
    # Run the function
    stramlit_trav_recomm()
    
    # Assertions to ensure UI components were called
    mock_st.title.assert_called_once_with("🏨 Travel Recommendations")
    mock_st.dataframe.assert_called_once()
    mock_st.bar_chart.assert_called_once()

@patch("test.st")
def test_streamlit_ui_no_model_loaded(mock_st):
    """Test UI behavior when the model is not in session state."""
    from test import stramlit_trav_recomm
    
    # Model is missing from session state
    mock_st.session_state = {}
    
    stramlit_trav_recomm()
    
    mock_st.error.assert_called_once_with("Model is not loaded. Please restart the application.")
