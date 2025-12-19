import os
import mlflow
import mlflow.sklearn  # Usaremos este para todo
import xgboost as xgb
import matplotlib.pyplot as plt
import optuna
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, f1_score
from .data_preprocess import get_processed_data


load_dotenv()

# Configuración
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "mlruns")
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "Telco_Churn_Project")
DATA_PATH = os.getenv("DATA_PATH", "data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

def objective(trial, X_train, X_test, y_train, y_test):
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'booster': 'gbtree',
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 10.0),
    }

    with mlflow.start_run(nested=True):
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)
        
        preds = model.predict(X_test)
        f1 = f1_score(y_test, preds)
        
        mlflow.log_params(params)
        mlflow.log_metric("f1_score", f1)
        
        return f1

def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    # Cargar datos y scaler
    X_train, X_test, y_train, y_test, scaler = get_processed_data(DATA_PATH)
    
    with mlflow.start_run(run_name="XGBoost_Optuna_Optimization"):
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: objective(trial, X_train, X_test, y_train, y_test), n_trials=10)
        
        print("Mejores params:", study.best_params)
        mlflow.log_params(study.best_params)
        mlflow.log_metric("best_f1_score", study.best_value)
        
        # Entrenar modelo final con los mejores parámetros
        best_model = xgb.XGBClassifier(**study.best_params)
        best_model.fit(X_train, y_train)
        
        # Usamos mlflow.sklearn porque XGBClassifier es un wrapper de sklearn.
        # Esto evita el error "_estimator_type undefined".
        mlflow.sklearn.log_model(best_model, "best_model")
        
        # Guardar también el Scaler
        mlflow.sklearn.log_model(scaler, "scaler")
        
        print("✅ Modelo y Scaler registrados correctamente en MLFlow.")

        plt.figure(figsize=(10, 8))
        xgb.plot_importance(best_model, max_num_features=10) 
        plt.title("Feature Importance")
        plt.savefig("feature_importance.png")
        plt.close() # Cierra la figura para liberar memoria
        
        mlflow.log_artifact("feature_importance.png")
        print("✅ Gráfico de interpretabilidad guardado.")
        
        if os.path.exists("feature_importance.png"):
            os.remove("feature_importance.png")

        print("✅ Modelo, Scaler y Gráficos registrados correctamente en MLFlow.")

if __name__ == "__main__":
    main()