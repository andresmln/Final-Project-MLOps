import pytest
import pandas as pd
from mylib.data_preprocess import clean_data

# Define a sample of "raw" dirty data like what you get from the CSV
@pytest.fixture
def raw_data():
    return pd.DataFrame({
        "gender": ["Male", "Female"],
        "SeniorCitizen": [0, 1],
        "Partner": ["Yes", "No"],
        "Dependents": ["No", "Yes"],
        "tenure": [1, 12],
        "PhoneService": ["No", "Yes"],
        "MultipleLines": ["No phone service", "No"],
        "InternetService": ["DSL", "Fiber optic"],
        "OnlineSecurity": ["No", "Yes"],
        "OnlineBackup": ["Yes", "No"],
        "DeviceProtection": ["No", "Yes"],
        "TechSupport": ["No", "Yes"],
        "StreamingTV": ["No", "Yes"],
        "StreamingMovies": ["No", "Yes"],
        "Contract": ["Month-to-month", "One year"],
        "PaperlessBilling": ["Yes", "No"],
        "PaymentMethod": ["Electronic check", "Mailed check"],
        "MonthlyCharges": [29.85, 56.95],
        "TotalCharges": ["29.85", " "],  # Includes a dirty empty string!
        "Churn": ["No", "Yes"]
    })

def test_clean_data_structure(raw_data):
    """Test that clean_data returns a valid DataFrame structure."""
    df_clean = clean_data(raw_data)
    
    # Check that it's still a dataframe
    assert isinstance(df_clean, pd.DataFrame)
    # Check that we haven't lost all rows
    assert len(df_clean) > 0

def test_total_charges_conversion(raw_data):
    """Test that 'TotalCharges' is converted from Object (string) to Numeric."""
    df_clean = clean_data(raw_data)
    
    # It should be a float or int now, not an object
    assert pd.api.types.is_numeric_dtype(df_clean["TotalCharges"])
    
    # The empty string row should have been dropped or handled
    # In your logic, you coerce errors='coerce' and dropna()
    assert df_clean["TotalCharges"].isnull().sum() == 0

def test_categorical_encoding_preparation(raw_data):
    """Test that categorical columns are preserved for encoding steps."""
    df_clean = clean_data(raw_data)
    assert "gender" in df_clean.columns
    assert "InternetService" in df_clean.columns