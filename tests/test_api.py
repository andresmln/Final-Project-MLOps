from fastapi.testclient import TestClient
from api.main import app

def test_read_root():
    """Check if the API is alive."""
    # Using 'with' triggers the startup event!
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert response.json() == {"message": "Telco Churn API is running"}

def test_predict_valid_input():
    """Test a valid prediction request returns 200 and correct JSON structure."""
    payload = {
        "gender": "Female", 
        "SeniorCitizen": "No", 
        "Partner": "Yes", 
        "Dependents": "No", 
        "tenure": 24, 
        "PhoneService": "Yes", 
        "MultipleLines": "No", 
        "InternetService": "DSL", 
        "OnlineSecurity": "No", 
        "OnlineBackup": "Yes", 
        "DeviceProtection": "No", 
        "TechSupport": "No", 
        "StreamingTV": "No", 
        "StreamingMovies": "No", 
        "Contract": "Month-to-month", 
        "PaperlessBilling": "Yes", 
        "PaymentMethod": "Electronic check", 
        "MonthlyCharges": 70.0, 
        "TotalCharges": 1680.0
    }
    
    # Using 'with' guarantees models are loaded
    with TestClient(app) as client:
        response = client.post("/predict", json=payload)
        
        # 1. Check Status Code (This was failing with 500)
        assert response.status_code == 200
        
        # 2. Check Response Content
        data = response.json()
        assert "churn_prediction" in data
        assert "churn_probability" in data
        
        # 3. Check Data Types
        assert isinstance(data["churn_prediction"], int)
        assert isinstance(data["churn_probability"], float)

def test_predict_invalid_input():
    """Test that missing fields cause a 422 Unprocessable Entity error."""
    with TestClient(app) as client:
        response = client.post("/predict", json={})
        assert response.status_code == 422