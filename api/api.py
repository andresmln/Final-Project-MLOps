from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from mylib.logic import load_model_and_scaler, churn_prediction

app = FastAPI(title="Telco Churn Prediction API")

# Variables globales
model = None
scaler = None

class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: str

@app.on_event("startup")
def startup_event():
    global model, scaler
    model, scaler = load_model_and_scaler()
    if model is None:
        print("⚠️ Advertencia: No se pudo cargar el modelo. Las predicciones fallarán.")
    else:
        print("✅ API lista con Modelo y Scaler cargados.")

@app.get("/")
def home():
    return {"message": "Telco Churn API is running"}

@app.post("/predict")
def predict(customer: CustomerData):
    global model, scaler
    if not model or not scaler:
        raise HTTPException(status_code=500, detail="Model not loaded in server")
    
    try:
        # Delegamos la lógica a la función pura
        result, probability = churn_prediction(customer.dict(), model, scaler)
        return {
            "churn_prediction": result,
            "churn_probability": probability
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error en predicción: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)