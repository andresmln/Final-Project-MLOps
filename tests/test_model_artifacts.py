import os
import pytest
import joblib
import pandas as pd

# The list of files that MUST exist for the API to work
REQUIRED_ARTIFACTS = [
    "api/models_local/model.joblib",         # Champion Model (XGBoost)
    "api/models_local/shadow_model.joblib",  # Challenger Model (TabNet)
    "api/models_local/scaler.joblib",        # Scaler
    "api/models_local/feature_names.joblib", # OneHotEncoder
    "api/models_local/threshold.joblib"       
]

# Define a single valid raw input sample
VALID_RAW_INPUT = {
    "gender": "Female", 
    "SeniorCitizen": 0, 
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

def test_artifacts_exist():
    """
    Verifies that the physical model files exist on disk.
    """
    missing_files = []
    for filename in REQUIRED_ARTIFACTS:
        if not os.path.exists(filename):
            missing_files.append(filename)
    
    if len(missing_files) == len(REQUIRED_ARTIFACTS):
        pytest.skip("No artifacts found. Run 'python mylib/train.py' first.")
    
    assert not missing_files, f"Missing critical artifacts: {missing_files}"

def test_load_artifacts():
    """
    Sanity check: Try to actually load the models with joblib 
    to ensure they are not corrupted files.
    """
    # 1. Check Champion Model
    model_path = "api/models_local/model.joblib"
    if os.path.exists(model_path):
        try:
            model = joblib.load(model_path)
            assert hasattr(model, "predict"), "Champion model invalid"
        except Exception as e:
            pytest.fail(f"Failed to load Champion Model: {e}")

    # 2. Check Shadow Model
    shadow_path = "api/models_local/shadow_model.joblib"
    if os.path.exists(shadow_path):
        try:
            shadow_model = joblib.load(shadow_path)
            assert hasattr(shadow_model, "predict"), "Shadow model invalid"
        except Exception as e:
            pytest.fail(f"Failed to load Shadow Model: {e}")

def get_preprocessed_input():
    """
    LOCAL HELPER: Mimics the API preprocessing logic.
    We do NOT use data_preprocess.py here because we need inference logic 
    (transform, not fit) and reindexing.
    """
    base_path = "api/models_local"
    scaler = joblib.load(os.path.join(base_path, "scaler.joblib"))
    feature_names = joblib.load(os.path.join(base_path, "feature_names.joblib"))

    df = pd.DataFrame([VALID_RAW_INPUT])
    
    binary_mapping = {'Yes': 1, 'No': 0, 'Female': 1, 'Male': 0}
    for col in ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'gender']:
        if col in df.columns:
            df[col] = df[col].map(binary_mapping).fillna(0)
    
    df = pd.get_dummies(df)
    
    # CRITICAL: Reindex to match training schema (The step data_preprocess lacks)
    df = df.reindex(columns=feature_names, fill_value=0)
    
    numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    df[numeric_cols] = scaler.transform(df[numeric_cols])
    
    return df

def test_champion_model_prediction():
    """
    Verifies the Champion Model (XGBoost) prediction logic.
    """
    if not os.path.exists("api/models_local/model.joblib"):
        pytest.skip("Champion model not found.")

    model = joblib.load("api/models_local/model.joblib")
    df = get_preprocessed_input()
    
    # XGBoost/Sklearn accepts DataFrame
    prediction = model.predict(df)
    probability = model.predict_proba(df)
    
    assert len(prediction) == 1
    assert prediction[0] in [0, 1]
    assert 0.0 <= probability[0][1] <= 1.0

def test_shadow_model_prediction():
    """
    Verifies the Shadow Model (TabNet) prediction logic.
    """
    if not os.path.exists("api/models_local/shadow_model.joblib"):
        pytest.skip("Shadow model not found.")

    shadow_model = joblib.load("api/models_local/shadow_model.joblib")
    df = get_preprocessed_input()
    
    # CRITICAL: TabNet expects NumPy array (.values), not DataFrame
    try:
        probability = shadow_model.predict_proba(df.values)
    except Exception as e:
        pytest.fail(f"Shadow model failed inference: {e}")
    
    assert len(probability) == 1
    assert 0.0 <= probability[0][1] <= 1.0