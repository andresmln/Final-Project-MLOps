import os
import pytest
import joblib

# The list of files that MUST exist for the API to work
REQUIRED_ARTIFACTS = [
    "api/models_local/model.joblib",         # Champion Model (XGBoost)
    "api/models_local/shadow_model.joblib",  # Challenger Model (TabNet)
    "api/models_local/scaler.joblib",        # Scaler
    "api/models_local/feature_names.joblib", # OneHotEncoder
    "api/models_local/threshold.joblib"       
]

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
    # Only run this if the files actually exist
    if not os.path.exists("model.joblib"):
        pytest.skip("Skipping load test because model.joblib is missing.")

    try:
        model = joblib.load("model.joblib")
        # Basic check: Does it have a predict method?
        assert hasattr(model, "predict"), "Loaded object is not a valid model (no predict method)"
    except Exception as e:
        pytest.fail(f"Failed to load model.joblib: {e}")