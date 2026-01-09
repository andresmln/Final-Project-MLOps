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

# Define a single valid raw input sample (Representative of what users send)
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
    This ensures that train.py and train_shadow.py ran successfully.
    """
    missing_files = []
    
    for filename in REQUIRED_ARTIFACTS:
        if not os.path.exists(filename):
            missing_files.append(filename)
    
    # If ALL files are missing, we might be in a fresh environment (pre-training).
    # In CI/CD, this should fail, but locally we might want to skip.
    if len(missing_files) == len(REQUIRED_ARTIFACTS):
        pytest.skip("No artifacts found. Run 'python mylib/train.py' first.")
    
    # If some are missing but not all, that's a broken build!
    assert not missing_files, f"Missing critical artifacts: {missing_files}"

def test_load_artifacts():
    """
    Sanity check: Try to actually load the models with joblib 
    to ensure they are not corrupted files.
    """
    # FIX: Use the correct path!
    model_path = "api/models_local/model.joblib"

    if not os.path.exists(model_path):
        pytest.skip(f"Skipping load test because {model_path} is missing.")

    try:
        model = joblib.load(model_path)
        # Basic check: Does it have a predict method?
        assert hasattr(model, "predict"), "Loaded object is not a valid model (no predict method)"
    except Exception as e:
        pytest.fail(f"Failed to load model.joblib: {e}")

def test_model_prediction_pipeline():
    """
    Component Test: Verifies that the full artifact chain (Scaler -> Schema -> Model)
    correctly transforms raw data into a prediction.
    """
    base_path = "api/models_local"
    
    # 1. Load ALL required artifacts
    try:
        model = joblib.load(os.path.join(base_path, "model.joblib"))
        scaler = joblib.load(os.path.join(base_path, "scaler.joblib"))
        feature_names = joblib.load(os.path.join(base_path, "feature_names.joblib"))
    except FileNotFoundError:
        pytest.skip("Artifacts not found. Run training first.")

    # 2. Simulate the Preprocessing (Mirrors logic in api/main.py)
    df = pd.DataFrame([VALID_RAW_INPUT])
    
    # Manual Binary Mapping (Simulating api/main.py logic)
    binary_mapping = {'Yes': 1, 'No': 0, 'Female': 1, 'Male': 0}
    # Note: We must handle columns that exist in input
    for col in ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'gender']:
        if col in df.columns:
            df[col] = df[col].map(binary_mapping).fillna(0)
    
    # One-Hot Encoding
    df = pd.get_dummies(df)
    
    # 2b. CRITICAL STEP: Reindex to match the training schema
    # This is what data_preprocess.py DOES NOT DO. 
    # It forces the single row to have the exact same columns as the training set.
    df = df.reindex(columns=feature_names, fill_value=0)
    
    # Scale numerical columns
    numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    df[numeric_cols] = scaler.transform(df[numeric_cols])
    
    # 3. Run Inference (The actual test)
    prediction = model.predict(df)
    probability = model.predict_proba(df)
    
    # 4. Assertions
    assert len(prediction) == 1, "Model did not return a prediction"
    assert prediction[0] in [0, 1], "Prediction must be binary"
    assert 0.0 <= probability[0][1] <= 1.0, "Probability must be valid"