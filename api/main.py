import os
import joblib
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, Response
from contextlib import asynccontextmanager
from pydantic import BaseModel
from prometheus_client import generate_latest
from enum import Enum

# IMPORT THE MONITORING LOGIC
from api.monitoring import (
    PREDICTION_COUNTER, 
    CHURN_PROBABILITY_GAUGE, 
    simulate_drift
)

# ==========================================
# 1. DATA MODELS & ENUMS
# ==========================================
class GenderEnum(str, Enum):
    male = "Male"
    female = "Female"

class YesNoEnum(str, Enum):
    yes = "Yes"
    no = "No"

class InternetServiceEnum(str, Enum):
    dsl = "DSL"
    fiber = "Fiber optic"
    no = "No"

class ContractEnum(str, Enum):
    month = "Month-to-month"
    one_year = "One year"
    two_year = "Two year"

class PaymentMethodEnum(str, Enum):
    electronic = "Electronic check"
    mailed = "Mailed check"
    transfer = "Bank transfer (automatic)"
    card = "Credit card (automatic)"

class YesNoPhoneEnum(str, Enum):
    yes = "Yes"
    no = "No"
    no_phone = "No phone service"

class YesNoInternetEnum(str, Enum):
    yes = "Yes"
    no = "No"
    no_internet = "No internet service"

class CustomerData(BaseModel):
    gender: GenderEnum
    SeniorCitizen: YesNoEnum
    Partner: YesNoEnum
    Dependents: YesNoEnum
    tenure: int
    PhoneService: YesNoEnum
    MultipleLines: YesNoPhoneEnum
    InternetService: InternetServiceEnum
    OnlineSecurity: YesNoInternetEnum
    OnlineBackup: YesNoInternetEnum
    DeviceProtection: YesNoInternetEnum
    TechSupport: YesNoInternetEnum
    StreamingTV: YesNoInternetEnum
    StreamingMovies: YesNoInternetEnum
    Contract: ContractEnum
    PaperlessBilling: YesNoEnum
    PaymentMethod: PaymentMethodEnum
    MonthlyCharges: float
    TotalCharges: float

# ==========================================
# 2. ARTIFACTS LOADING
# ==========================================
artifacts = {
    "model": None,
    "shadow_model": None,
    "scaler": None,
    "feature_names": None,
    "threshold": 0.5
}

def load_artifacts():
    try:
        # 1. Get the directory where this script (main.py) is actually located
        current_dir = os.path.dirname(os.path.abspath(__file__))

        # 2. Construct the absolute path to the 'models_local' folder inside 'api'
        # This ensures it works whether you run 'pytest' from root or 'uvicorn' from api/
        default_path = os.path.join(current_dir, "models_local")

        # 3. Use Environment Variable if set (for Docker), otherwise use the path we just built
        base_path = os.getenv("ARTIFACT_PATH", default_path)
        
        print(f"📂 Loading artifacts from: {base_path}")
        
        artifacts["model"] = joblib.load(os.path.join(base_path, "model.joblib"))
        artifacts["shadow_model"] = joblib.load(os.path.join(base_path, "shadow_model.joblib"))
        artifacts["scaler"] = joblib.load(os.path.join(base_path, "scaler.joblib"))
        artifacts["feature_names"] = joblib.load(os.path.join(base_path, "feature_names.joblib"))
        artifacts["threshold"] = joblib.load(os.path.join(base_path, "threshold.joblib"))
        
        print("✅ Artifacts loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading artifacts: {e}")
        # Optional: Print current working directory to help debug
        print(f"PWD: {os.getcwd()}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    The new standard way to handle startup/shutdown in FastAPI.
    """
    # STARTUP: Load models
    load_artifacts()
    yield
    # SHUTDOWN: Clean up
    artifacts.clear()

app = FastAPI(title="Telco Churn Prediction API", lifespan=lifespan)

# ==========================================
# 3. PREPROCESSING
# ==========================================
def preprocess_input(data: dict) -> pd.DataFrame:
    df = pd.DataFrame([data])
    
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = df['TotalCharges'].fillna(0)

    binary_mapping = {
        'Yes': 1, 'No': 0, 
        'True': 1, 'False': 0, 
        'Female': 1, 'Male': 0,
        '1': 1, '0': 0
    }
    
    binary_cols = ['Partner', 'Dependents', 'PhoneService', 
                   'PaperlessBilling', 'gender', 'SeniorCitizen']
    
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].map(binary_mapping).fillna(0).astype(int)

    df = pd.get_dummies(df)
    
    # Align with Schema
    required_cols = artifacts["feature_names"]
    df = df.reindex(columns=required_cols, fill_value=0)
    
    # Scale
    numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    df[numeric_cols] = artifacts["scaler"].transform(df[numeric_cols])
    
    return df


# ==========================================
# 4. SHADOW DEPLOYMENT LOGIC
# ==========================================
def run_shadow_inference(input_df: pd.DataFrame):
    """
    Runs TabNet in the background. logs result but does NOT return it to user.
    """
    if artifacts["shadow_model"]:
        try:
            # TabNet expects numpy array
            shadow_prob = artifacts["shadow_model"].predict_proba(input_df.values)[:, 1][0]
            print(f"👻 [SHADOW MODE] TabNet Prediction: {shadow_prob:.4f}")
            # In real life, you would save this to a database to compare later
        except Exception as e:
            print(f"⚠️ Shadow inference failed: {e}")



# ==========================================
# 5. ENDPOINTS
# ==========================================

@app.on_event("startup")
def startup_event():
    load_artifacts()

@app.get("/")
def home():
    return {"message": "Telco Churn API is running"}

@app.get("/metrics")
def metrics():
    """
    Endpoint for Prometheus. 
    1. Runs the drift simulation logic.
    2. Returns all metrics in plain text format.
    """
    simulate_drift()
    return Response(generate_latest(), media_type="text/plain")

@app.post("/predict")
def predict(customer: CustomerData):
    if not artifacts["model"]:
        raise HTTPException(status_code=500, detail="Model not loaded.")
    
    try:
        # Increment Counter
        PREDICTION_COUNTER.inc()
        
        input_df = preprocess_input(customer.model_dump())

        run_shadow_inference(input_df)

        prob = artifacts["model"].predict_proba(input_df)[:, 1][0]
        prediction = 1 if prob >= artifacts["threshold"] else 0
        
        # Update Gauge
        CHURN_PROBABILITY_GAUGE.set(prob)
        
        return {
            "churn_prediction": int(prediction),
            "churn_probability": float(prob),
            "threshold_used": float(artifacts["threshold"])
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)