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

    df['SeniorCitizen'] = pd.to_numeric(df['SeniorCitizen'], errors='coerce').fillna(0).astype(int)
    
    # 1. Manual Binary Mapping
    binary_mapping = {'Yes': 1, 'No': 0, 'True': 1, 'False': 0, 'Female': 1, 'Male': 0}
    binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn', 'gender']
    
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].map(binary_mapping).fillna(0).astype(int)

    # 2. One-Hot Encoding
    df = pd.get_dummies(df, drop_first=True, dtype=int)
        
    return df

def get_processed_data(filepath, target_col='Churn', test_size=0.2, val_size=0.2, seed=42):
    """
    Returns: X_train, X_val, X_test, y_train, y_val, y_test, scaler, feature_names
    """
    # 1. Load & Clean
    df = load_data(filepath)
    df = clean_data(df)
    df = encode_data(df)
    
    # 2. Split X/y
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    # 3. First Split: Separate out the FINAL Test Set (e.g. 20%)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    
    # 4. Second Split: Separate Train and Validation from the remaining data
    # We want Val to be 20% of the TOTAL original data.
    # The 'temp' data is 80% of the total.
    # So we need 0.2 / 0.8 = 0.25 (25%) of the temp data.
    relative_val_size = val_size / (1 - test_size)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=relative_val_size, random_state=seed, stratify=y_temp
    )
    
    # 5. Scale Numerical Cols (Fit on Train ONLY)
    numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    scaler = StandardScaler()
    
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_val[numeric_cols] = scaler.transform(X_val[numeric_cols])    # Transform Val
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])  # Transform Test
    
    # 6. Capture feature names
    feature_names = X_train.columns.tolist()
    
    print(f"✅ Data Processed: Train {X_train.shape}, Val {X_val.shape}, Test {X_test.shape}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, scaler, feature_names