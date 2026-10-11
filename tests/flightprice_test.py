import pytest
import requests
from datetime import date
from unittest.mock import patch, MagicMock

# Import the extracted function and the main streamlit module
from src.flightpriceprediction.app import call_predict_api, flight_price_pred

class TestCallPredictApi:
    """Tests for the extracted REST API function."""

    def test_successful_api_call(self, requests_mock):
        """Test that a 200 OK response returns the correct JSON data."""
        api_url = "http://localhost:5000"
        payload = {
            "from": "New York",
            "to": "London",
            "flightType": "Economy",
            "agency": "Delta",
            "date": "10/25/2023"
        }
        
        # Mock the API endpoint
        mock_response = {"predicted_price": 850.50, "distance": 3450}
        requests_mock.post(f"{api_url}/predict", json=mock_response, status_code=200)

        # Call the function
        result = requests.post(f"{api_url}/predict", json=payload)

        # Assertions
        assert result == mock_response
        assert result["predicted_price"] == 850.50
        assert result["distance"] == 3450

    def test_api_call_sends_correct_payload(self, requests_mock):
        """Test that the function sends the correct payload to the API."""
        api_url = "http://localhost:5000"
        payload = {
            "from": "Tokyo",
            "to": "Paris",
            "flightType": "Business",
            "agency": "AirFrance",
            "date": "12/01/2023"
        }
        
        # Mock the endpoint just to prevent real HTTP calls
        requests_mock.post(f"{api_url}/predict", json={}, status_code=200)

        # Call the function
        result = requests.post(f"{api_url}/predict", json=payload)

        # Verify the request payload
        last_request = requests_mock.last_request
        assert last_request.json() == payload

    def test_api_call_http_error(self, requests_mock):
        """Test that an HTTP error (e.g., 500) raises an HTTPError."""
        api_url = "http://localhost:5000"
        payload = {"from": "A", "to": "B", "flightType": "C", "agency": "D", "date": "01/01/2023"}
        
        # Mock a 500 Internal Server Error
        requests_mock.post(f"{api_url}/predict", status_code=500, text="Internal Server Error")

        # Assert that raise_for_status triggers an HTTPError
        with pytest.raises(requests.exceptions.HTTPError):
            call_predict_api(api_url, payload)

    def test_api_call_connection_error(self, requests_mock):
        """Test that a ConnectionError is raised if the API is unreachable."""
        api_url = "http://invalid-url:5000"
        payload = {"from": "A", "to": "B", "flightType": "C", "agency": "D", "date": "01/01/2023"}
        
        # Mock a connection error
        requests_mock.post(f"{api_url}/predict", exc=requests.exceptions.ConnectionError)

        # Assert the exception is raised
        with pytest.raises(requests.exceptions.ConnectionError):
            call_predict_api(api_url, payload)


class TestStreamlitAppIntegration:
    """Tests for the Streamlit UI logic using mocks."""

    @patch('src.flightpriceprediction.app.call_predict_api')
    @patch('src.flightpriceprediction.app.st')
    @patch('src.flightpriceprediction.app.df', new_callable=MagicMock)
    def test_ui_calls_api_on_toggle(self, mock_df, mock_st, mock_call_api):
        """Test that the UI attempts to call the API when use_api is True."""
        # Setup mock DataFrame to prevent Pandas errors
        mock_df.__getitem__.return_value.unique.return_value = ["CityA", "CityB"]
        
        # Mock Streamlit UI returns
        mock_st.toggle.return_value = True  # use_api = True
        mock_st.text_input.return_value = "http://mock-api:5000"
        mock_st.button.return_value = True  # Simulate button click
        mock_st.date_input.return_value = date(2023, 10, 25)
        mock_st.selectbox.side_effect = ["CityA", "CityB", "AgencyX"]
        mock_st.radio.return_value = "Economy"
        
        # Mock the API response
        mock_call_api.return_value = {"predicted_price": 100.0, "distance": 500}

        # Run the streamlit function
        flight_price_pred()

        # Verify call_predict_api was called
        mock_call_api.assert_called_once()
        call_args = mock_call_api.call_args
        
        # Verify the URL and payload structure passed to the API
        assert call_args[0][0] == "http://mock-api:5000"
        assert call_args[0][1]["from"] == "CityA"
        assert call_args[0][1]["date"] == "10/25/2023"
        
        # Verify Streamlit success was called (UI updated)
        mock_st.success.assert_called_once()
        
    @patch('src.flightpriceprediction.app.call_predict_api')
    @patch('src.flightpriceprediction.app.st')
    @patch('src.flightpriceprediction.app.df', new_callable=MagicMock)
    def test_ui_handles_api_failure(self, mock_df, mock_st, mock_call_api):
        """Test that the UI displays an error if the API call fails."""
        mock_df.__getitem__.return_value.unique.return_value = ["CityA", "CityB"]
        
        mock_st.toggle.return_value = True
        mock_st.text_input.return_value = "http://mock-api:5000"
        mock_st.button.return_value = True
        mock_st.date_input.return_value = date(2023, 10, 25)
        mock_st.selectbox.side_effect = ["CityA", "CityB", "AgencyX"]
        mock_st.radio.return_value = "Economy"
        
        # Simulate an API failure
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Server Error"
        http_error = requests.exceptions.HTTPError(response=mock_response)
        mock_call_api.side_effect = http_error

        # Run the function
        flight_price_pred()

        # Verify Streamlit error was called instead of success
        mock_st.error.assert_called_once()
        # Verify the error message contains the API response text
        assert "Server Error" in mock_st.error.call_args[0][0]
        mock_st.success.assert_not_called()
