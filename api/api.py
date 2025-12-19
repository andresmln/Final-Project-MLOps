from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import mlflow.sklearn
import os
import uvicorn
from mylib.data_preprocess import clean_data, encode_categorical_data

app = FastAPI(title="Telco Churn Prediction API")

# Rutas globales para cargar artefactos
# NOTA: En un entorno real, esto se cargaría dinámicamente o desde un Model Registry.
# Para la práctica, buscaremos el último run exitoso en mlruns o cargamos localmente si lo exportaste.
LOGGED_MODEL_URI = "models:/Telco_Churn_Project/Production" # Si usas Model Registry
# O alternativamente, ruta local si acabas de entrenar:
MLRUNS_DIR = "mlruns"

# Variables globales para modelo y scaler
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
    TotalCharges: str  # Viene como string a veces, lo limpiamos dentro

@app.on_event("startup")
def load_artifacts():
    global model, scaler
    try:
        print("Cargando modelo y scaler más recientes...")
        # Buscamos el último experimento exitoso para cargar sus artefactos
        # (Este es un truco para la práctica si no usas MLflow Model Registry explícito)
        experiment = mlflow.get_experiment_by_name("Telco_Churn_Project")
        if experiment is None:
             raise Exception("No se encontró el experimento en MLFlow")
        
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"],
            max_results=1
        )
        
        if runs.empty:
            raise Exception("No hay runs registrados todavía.")
            
        last_run_id = runs.iloc[0].run_id
        print(f"Cargando artefactos del Run ID: {last_run_id}")
        
        # Cargar Modelo XGBoost (usamos sklearn flavor porque así lo guardamos)
        model = mlflow.sklearn.load_model(f"runs:/{last_run_id}/best_model")
        # Cargar Scaler
        scaler = mlflow.sklearn.load_model(f"runs:/{last_run_id}/scaler")
        
        print("✅ Artefactos cargados correctamente.")
    except Exception as e:
        print(f"❌ Error cargando modelo: {e}")
        # No matamos la app, pero las predicciones fallarán si no se arregla

@app.get("/")
def home():
    return {"message": "Telco Churn API is running"}

@app.post("/predict")
def predict_churn(customer: CustomerData):
    global model, scaler
    if not model or not scaler:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # 1. Convertir JSON a DataFrame
        input_data = customer.dict()
        df = pd.DataFrame([input_data])
        
        # 2. Limpieza (Reutilizamos funciones de mylib)
        df = clean_data(df)
        
        # 3. Encoding (Reutilizamos lógica)
        df = encode_categorical_data(df)
        
        # 4. Escalado (Usamos el scaler cargado, NO uno nuevo)
        numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
        # Aseguramos que las columnas existan y estén en orden
        df[numeric_cols] = scaler.transform(df[numeric_cols])
        
        # 5. Predecir
        # Asegurar que el orden de columnas coincida con el entrenamiento
        # (XGBoost es sensible al orden, obtenemos las features del modelo)
        prediction = model.predict(df)
        probability = model.predict_proba(df)[:, 1]
        
        result = "Yes" if prediction[0] == 1 else "No"
        
        return {
            "churn_prediction": result,
            "churn_probability": float(probability[0])
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing data: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)