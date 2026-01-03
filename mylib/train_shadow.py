import os
import mlflow
import mlflow.pytorch
import torch
import optuna
import joblib
import json
import numpy as np
import pandas as pd
from pytorch_tabnet.tab_model import TabNetClassifier
from dotenv import load_dotenv
from sklearn.metrics import average_precision_score, accuracy_score, roc_auc_score
from mylib.data_preprocess import get_processed_data

load_dotenv()

# Config
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "mlruns")
EXPERIMENT_NAME = "Telco_Churn_Shadow_Model" # Separate experiment for clarity
DATA_PATH = os.getenv("DATA_PATH", "archive/WA_Fn-UseC_-Telco-Customer-Churn.csv")

def objective(trial, X_train, X_test, y_train, y_test):
    """
    Optuna Objective for TabNet.
    Optimizes architecture (n_d, n_a) and training (lr, gamma).
    """
    # TabNet Hyperparameters
    n_da = trial.suggest_int('n_da', 8, 64, step=8) # n_d and n_a are usually same
    params = {
        'n_d': n_da, 
        'n_a': n_da,
        'n_steps': trial.suggest_int('n_steps', 3, 10),
        'gamma': trial.suggest_float('gamma', 1.0, 2.0),
        'lambda_sparse': trial.suggest_float('lambda_sparse', 1e-4, 1e-2, log=True),
        'optimizer_fn': torch.optim.Adam,
        'optimizer_params': dict(lr=trial.suggest_float('learning_rate', 1e-3, 1e-1, log=True)),
        'mask_type': 'entmax', 
        'verbose': 0
    }
    
    # Training parameters
    batch_size = trial.suggest_categorical('batch_size', [256, 512, 1024])
    virtual_batch_size = 128
    
    with mlflow.start_run(nested=True):
        model = TabNetClassifier(**params)
        
        # TabNet expects NumPy arrays
        model.fit(
            X_train=X_train.values, y_train=y_train.values,
            eval_set=[(X_train.values, y_train.values), (X_test.values, y_test.values)],
            eval_name=['train', 'valid'],
            eval_metric=['auc'],
            max_epochs=20, # Low epochs for faster tuning
            patience=5,
            batch_size=batch_size,
            virtual_batch_size=virtual_batch_size,
            num_workers=0,
            drop_last=False
        )
        
        # Evaluate
        preds_prob = model.predict_proba(X_test.values)[:, 1]
        score = average_precision_score(y_test, preds_prob)
        
        mlflow.log_params(params)
        mlflow.log_metric("pr_auc_optimization", score)
        
        return score

def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    # 1. Load Data (Reusing your central logic)
    print("📥 Loading and Processing Data...")
    X_train, X_test, y_train, y_test, scaler, feature_names = get_processed_data(DATA_PATH)
    
    with mlflow.start_run(run_name="Shadow_Model_Pipeline") as run:
        
        # ---------------------------------------------------------
        # PHASE A: HYPERPARAMETER OPTIMIZATION
        # ---------------------------------------------------------
        print("🔍 Optimizing TabNet Hyperparameters...")
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: objective(trial, X_train, X_test, y_train, y_test), n_trials=10)
        
        print("🏆 Best Params:", study.best_params)
        mlflow.log_params(study.best_params)
        
        # ---------------------------------------------------------
        # PHASE B: FINAL TRAINING
        # ---------------------------------------------------------
        print("⚙️ Training Final Shadow Model...")
        
        # Reconstruct params from study (TabNet requires specific structure)
        n_da = study.best_params['n_da']
        best_params = {
            'n_d': n_da, 
            'n_a': n_da,
            'n_steps': study.best_params['n_steps'],
            'gamma': study.best_params['gamma'],
            'lambda_sparse': study.best_params['lambda_sparse'],
            'optimizer_fn': torch.optim.Adam,
            'optimizer_params': dict(lr=study.best_params['learning_rate']),
            'mask_type': 'entmax',
            'verbose': 1
        }
        
        clf_tabnet = TabNetClassifier(**best_params)
        
        clf_tabnet.fit(
            X_train=X_train.values, y_train=y_train.values,
            eval_set=[(X_train.values, y_train.values), (X_test.values, y_test.values)],
            eval_name=['train', 'valid'],
            eval_metric=['auc'],
            max_epochs=50, 
            patience=10,
            batch_size=study.best_params['batch_size'],
            virtual_batch_size=128,
            num_workers=0,
            drop_last=False
        )

        # ---------------------------------------------------------
        # PHASE C: EVALUATION & LOGGING
        # ---------------------------------------------------------
        preds_prob = clf_tabnet.predict_proba(X_test.values)[:, 1]
        auc = roc_auc_score(y_test, preds_prob)
        print(f"📊 Shadow Model AUC: {auc:.4f}")
        mlflow.log_metric("auc", auc)

        # ---------------------------------------------------------
        # PHASE D: ARTIFACT SERIALIZATION
        # ---------------------------------------------------------
        print("💾 Saving Artifacts...")
        
        # 1. Save ONLY the Shadow Model
        # We DO NOT save scaler/feature_names here to avoid overwriting the Main Model's files.
        # The Shadow Model piggybacks off the Main Model's preprocessing in the API.
        
        joblib.dump(clf_tabnet, "shadow_model.joblib")
        mlflow.log_artifact("shadow_model.joblib", artifact_path="shadow_model")
        
        print("✅ shadow_model.joblib saved successfully.")
        print("✅ Full Shadow Pipeline Completed.")

if __name__ == "__main__":
    main()