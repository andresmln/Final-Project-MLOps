import os
import mlflow
import mlflow.sklearn
import xgboost as xgb
import matplotlib.pyplot as plt
import optuna
import joblib
import json
import shap
import numpy as np
import pandas as pd
import time
from dotenv import load_dotenv
from sklearn.metrics import (
    average_precision_score, 
    precision_recall_curve, 
    accuracy_score, 
    confusion_matrix,
    ConfusionMatrixDisplay
)
from mylib.data_preprocess import get_processed_data

load_dotenv()



# Configuración
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "mlruns")
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "Telco_Churn_Model")
DATA_PATH = os.getenv("DATA_PATH", "archive/WA_Fn-UseC_-Telco-Customer-Churn.csv")
OUTPUT_DIR = "api/models_local"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def objective(trial, X_train, X_val, y_train, y_val):
    """
    METRIC 1: Threshold INDEPENDENT (PR-AUC)
    Used purely to find the best Hyperparameters.
    """
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'booster': 'gbtree',
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 10.0),
    }

    with mlflow.start_run(nested=True):
        model = xgb.XGBClassifier(**params, use_label_encoder=False)
        model.fit(X_train, y_train)
        
        # Predict Probabilities (Not Classes)
        y_proba = model.predict_proba(X_val)[:, 1]
        
        # Optimize based on PR-AUC (Average Precision)
        # This metric doesn't care about thresholds (0.5 vs 0.3), only ranking quality.
        score = average_precision_score(y_val, y_proba)
        
        mlflow.log_params(params)
        mlflow.log_metric("pr_auc_optimization", score)
        
        return score

def find_best_threshold(y_true, y_proba):
    """
    METRIC 2: Threshold DEPENDENT logic.
    Finds the threshold that maximizes F1-Score.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    # F1 = 2 * (P * R) / (P + R)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10) # Add epsilon to avoid div/0
    
    # Locate the index of the largest F1 score
    ix = np.argmax(f1_scores)
    best_thresh = thresholds[ix]
    best_f1 = f1_scores[ix]
    
    return best_thresh, best_f1

def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    # 1. Load Data
    X_train, X_val, X_test, y_train, y_val, y_test, scaler, feature_names = get_processed_data(DATA_PATH)
    
    with mlflow.start_run(run_name="Production_Candidate_Run") as run:
        
        # ---------------------------------------------------------
        # PHASE A: HYPERPARAMETER OPTIMIZATION (Threshold Independent)
        # ---------------------------------------------------------
        print("🔍 Optimizing Hyperparameters...")
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: objective(trial, X_train, X_val, y_train, y_val), n_trials=20)
        
        print("🏆 Best Params:", study.best_params)
        mlflow.log_params(study.best_params)
        
        # ---------------------------------------------------------
        # PHASE B: THRESHOLD SELECTION (Threshold Dependent)
        # ---------------------------------------------------------
        print("⚙️ Training Final Model...")
        best_model = xgb.XGBClassifier(**study.best_params, use_label_encoder=False)

        # START TIMER
        train_start = time.time()

        best_model.fit(X_train, y_train)

        # STOP TIMER
        train_end = time.time()
        training_time = train_end - train_start
        
        print(f"⏱️ Training Time: {training_time:.4f} seconds")
        mlflow.log_metric("training_time_seconds", training_time)
        
        # Get probabilities for Val set
        y_proba_val = best_model.predict_proba(X_val)[:, 1]
        
        # Calculate optimal threshold
        best_threshold, max_f1 = find_best_threshold(y_val, y_proba_val)
        
        print(f"🎯 Optimal Threshold found: {best_threshold:.4f} (Max F1: {max_f1:.4f})")
        mlflow.log_param("optimal_threshold", best_threshold)
        mlflow.log_metric("optimal_f1_score", max_f1)
        
        # ---------------------------------------------------------
        # PHASE C: LOGGING PRODUCTION METRICS
        # ---------------------------------------------------------
        # Apply the threshold to generate hard classes (0 or 1)
        y_pred_hard = (y_proba_val >= best_threshold).astype(int)
        
        final_acc = accuracy_score(y_val, y_pred_hard)
        mlflow.log_metric("production_accuracy", final_acc)
        
        ## METRIC.JSON PARA HUGGING FACE
        metrics_data = {
            "accuracy": float(final_acc),
            "f1_score": float(max_f1),
            "training_time_sec": float(training_time)
        }
        
        # Save directly to api/models_local
        metrics_path = os.path.join(OUTPUT_DIR, "metrics.json") # <--- CHANGED PATH
        with open(metrics_path, "w") as f:
            json.dump(metrics_data, f)
            
        print(f"✅ metrics.json saved to {metrics_path}")      
          
        # Log Confusion Matrix Plot
        cm = confusion_matrix(y_val, y_pred_hard)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Churn", "Churn"])
        disp.plot(cmap="Blues")
        plt.title(f"Confusion Matrix (Threshold: {best_threshold:.2f})")
        plt.savefig("confusion_matrix.png")
        plt.close()
        mlflow.log_artifact("confusion_matrix.png", artifact_path="plots")

        # ---------------------------------------------------------
        # PHASE D: ARTIFACT SERIALIZATION
        # ---------------------------------------------------------
        # 1. Save Model
        mlflow.sklearn.log_model(best_model, "model")
        # 1b. Save Model as joblib
        joblib.dump(best_model, os.path.join(OUTPUT_DIR, "model.joblib"))
        mlflow.log_artifact(os.path.join(OUTPUT_DIR, "model.joblib"), artifact_path="preprocessing")
        
        # 2. Save Preprocessing Artifacts
        joblib.dump(scaler, os.path.join(OUTPUT_DIR, "scaler.joblib"))
        joblib.dump(feature_names, os.path.join(OUTPUT_DIR, "feature_names.joblib"))
        joblib.dump(best_threshold, os.path.join(OUTPUT_DIR, "threshold.joblib"))
        
        mlflow.log_artifact(os.path.join(OUTPUT_DIR, "scaler.joblib"), artifact_path="preprocessing")
        mlflow.log_artifact(os.path.join(OUTPUT_DIR, "feature_names.joblib"), artifact_path="preprocessing")
        mlflow.log_artifact(os.path.join(OUTPUT_DIR, "threshold.joblib"), artifact_path="preprocessing")
        
        # Log the Processed Data for Auditing/Debugging
        # We save it to a temp file first, then upload it to MLflow
        processed_df = X_train.copy()
        processed_df['TARGET_CHURN'] = y_train
        processed_df.to_csv("processed_data_audit.csv", index=False)
        mlflow.log_artifact("processed_data_audit.csv", artifact_path="data_lineage")
        # Cleanup temp file
        if os.path.exists("processed_data_audit.csv"):
            os.remove("processed_data_audit.csv")
        
        # 3. Global Interpretability (SHAP)
        print("📊 Generating SHAP plot...")
        explainer = shap.TreeExplainer(best_model)
        shap_values = explainer.shap_values(X_val)
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_val, max_display=12, show=False, plot_type="dot")
        plt.title("SHAP Feature Importance")
        plt.tight_layout()
        plt.savefig("shap_summary.png", bbox_inches='tight', dpi=300)
        plt.close()
        mlflow.log_artifact("shap_summary.png", artifact_path="plots")
        
        # Cleanup
        ##for f in ["scaler.joblib", "feature_names.joblib", "threshold.joblib", "shap_summary.png", "confusion_matrix.png"]:
          #  if os.path.exists(f):
           #     os.remove(f)
        for f in ["shap_summary.png", "confusion_matrix.png", "processed_data_audit.csv"]:
            if os.path.exists(f):
                os.remove(f)

        print("✅ Full Pipeline Completed.")

if __name__ == "__main__":
    main()