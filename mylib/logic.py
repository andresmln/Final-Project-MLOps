import mlflow
import mlflow.sklearn
import pandas as pd
from .data_preprocess import clean_data, encode_categorical_data

def load_model_and_scaler(experiment_name="Telco_Churn_Project"):
    """
    Busca el último Run exitoso en MLFlow y carga el modelo y el scaler.
    Devuelve (model, scaler) o lanza una excepción si falla.
    """
    try:
        print(f"Buscando artefactos en el experimento: {experiment_name}")
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            return None, None
        
        # Obtener el último run
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"],
            max_results=1
        )
        
        if runs.empty:
            return None, None
            
        last_run_id = runs.iloc[0].run_id
        print(f"Cargando desde Run ID: {last_run_id}")
        
        # Cargar usando sklearn (ya que lo guardamos con ese flavor en train.py)
        model = mlflow.sklearn.load_model(f"runs:/{last_run_id}/best_model")
        scaler = mlflow.sklearn.load_model(f"runs:/{last_run_id}/scaler")
        
        return model, scaler
    except Exception as e:
        print(f"Error cargando artefactos: {e}")
        return None, None

def churn_prediction(customer_dict, model, scaler):
    """
    Recibe un diccionario con datos del cliente y devuelve la predicción.
    Realiza los mismos pasos que el entrenamiento: Clean -> Encode -> Scale -> Predict.
    """
    # 1. Convertir diccionario a DataFrame
    df = pd.DataFrame([customer_dict])
    
    # 2. Limpieza
    df = clean_data(df)
    
    # 3. Encoding
    df = encode_categorical_data(df)
    
    # 4. Escalado (Usando el scaler entrenado)
    numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    # Verificamos que las columnas existan antes de transformar
    if all(col in df.columns for col in numeric_cols):
        df[numeric_cols] = scaler.transform(df[numeric_cols])
    
    # 5. Predecir
    # XGBoost devuelve numpy arrays, extraemos el valor
    pred_label = model.predict(df)[0]
    pred_prob = model.predict_proba(df)[0][1]
    
    result = "Yes" if pred_label == 1 else "No"
    return result, float(pred_prob)