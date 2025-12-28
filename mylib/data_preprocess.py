import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

def load_data(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    return pd.read_csv(filepath)

def clean_data(df):
    """
    Standard cleaning: Duplicates check + TotalCharges fix + ID removal.
    """
    df = df.copy()
    
    # 1. Duplicates Check
    if df.duplicated().sum() > 0:
        print(f"Removed {df.duplicated().sum()} fully duplicate rows.")
        df = df.drop_duplicates()

    # 2. ID Integrity Check
    if 'customerID' in df.columns:
        if df['customerID'].duplicated().any():
            df = df.drop_duplicates(subset=['customerID'], keep='first')
        df = df.drop('customerID', axis=1)
        
    # 3. Fix TotalCharges
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
    
    return df

def encode_data(df):
    """
    Applies One-Hot Encoding.
    """
    df = df.copy()
    
    # 1. Manual Binary Mapping (Safe & Explicit)
    # Mapping these specifically guarantees 0/1 integers
    binary_mapping = {'Yes': 1, 'No': 0, 'True': 1, 'False': 0, 'Female': 1, 'Male': 0}
    binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn', 'gender']
    
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].map(binary_mapping).fillna(0).astype(int)

    # 2. One-Hot Encoding (Get Dummies)
    # drop_first=True prevents multicollinearity (Dummy Variable Trap)
    # dtype=int ensures we get 0/1 instead of True/False
    df = pd.get_dummies(df, drop_first=True, dtype=int)
        
    return df

def get_processed_data(filepath, target_col='Churn', test_size=0.2, seed=42):
    # 1. Load
    df = load_data(filepath)
    
    # 2. Clean
    df = clean_data(df)
    
    # 3. Encode (One-Hot)
    df = encode_data(df)
    
    # 4. Split X/y
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    # 5. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    
    # 6. Scale Numerical Cols
    # (Important: Scale AFTER split to avoid data leakage)
    numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    scaler = StandardScaler()
    
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])
    
    # 7. CRITICAL STEP FOR MLOPS: Capture the final column structure
    # We need this list to align future data (inference) to this exact shape
    feature_names = X_train.columns.tolist()
    
    return X_train, X_test, y_train, y_test, scaler, feature_names