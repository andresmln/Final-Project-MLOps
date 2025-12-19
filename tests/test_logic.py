import pytest
from mylib.logic import churn_prediction
from unittest.mock import MagicMock
import pandas as pd

def test_churn_prediction_structure():
    """
    Prueba que la función de lógica orquesta los pasos correctamente.
    Usamos Mocks para no necesitar un modelo real entrenado durante este test unitario.
    """
    # 1. Datos de entrada (Input Mock)
    customer_data = {
        "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
        "tenure": 1, "PhoneService": "No", "MultipleLines": "No phone service",
        "InternetService": "DSL", "OnlineSecurity": "No", "OnlineBackup": "Yes",
        "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
        "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check", "MonthlyCharges": 29.85, "TotalCharges": "29.85"
    }

    # 2. Crear Mocks para Modelo y Scaler
    mock_model = MagicMock()
    mock_model.predict.return_value = [1] # Simula que predice "Yes" (1)
    mock_model.predict_proba.return_value = [[0.2, 0.8]] # Simula 80% probabilidad
    
    mock_scaler = MagicMock()
    # Cuando el scaler recibe un DataFrame, devuelve el mismo (o numpy array)
    mock_scaler.transform.return_value = pd.DataFrame(
        {'tenure': [0.1], 'MonthlyCharges': [0.2], 'TotalCharges': [0.3]}
    )

    # 3. Ejecutar la función
    result, prob = churn_prediction(customer_data, mock_model, mock_scaler)

    # 4. Validaciones
    assert result == "Yes"
    assert prob == 0.8
    # Verificar que se llamó al scaler y al modelo
    mock_scaler.transform.assert_called_once()
    mock_model.predict.assert_called_once()