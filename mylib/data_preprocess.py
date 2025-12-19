import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os

def load_data(filepath):
    """Carga el dataset verificando que exista."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"El archivo no existe en: {filepath}")
    return pd.read_csv(filepath)

def clean_data(df):
    """
    Limpieza inicial de datos:
    1. Elimina customerID (no aporta información).
    2. Convierte TotalCharges a numérico y rellena nulos.
    """
    df = df.copy()
    
    # 1. Eliminar ID
    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)
        
    # 2. Corregir TotalCharges (hay espacios en blanco que deben ser 0)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(0)
    
    return df

def encode_categorical_data(df):
    """
    Transforma textos a números usando Label Encoding.
    Mantiene las columnas numéricas intactas.
    """
    df = df.copy()
    
    # Variables binarias manuales (opcional, pero ayuda a la claridad)
    yes_no_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
    for col in yes_no_cols:
        if col in df.columns:
            df[col] = df[col].map({'Yes': 1, 'No': 0})
            
    if 'gender' in df.columns:
        df['gender'] = df['gender'].map({'Female': 1, 'Male': 0})
        
    # Codificar el resto de categóricas (InternetService, Contract, etc.)
    le = LabelEncoder()
    for col in df.select_dtypes(include=['object', 'category']).columns:
        df[col] = le.fit_transform(df[col].astype(str))
        
    return df

def get_processed_data(filepath, target_col='Churn', test_size=0.2, seed=42):
    # 1. Cargar
    df = load_data(filepath)
    
    # 2. Limpiar
    df = clean_data(df)
    
    # 3. Codificar categóricas
    df = encode_categorical_data(df)
    
    # 4. Separar X e y
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    # 5. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    
    # 6. Escalar (AQUÍ ESTÁ LA CLAVE)
    numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    scaler = StandardScaler()
    
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])
    
    # CAMBIO: Devolvemos también el scaler
    return X_train, X_test, y_train, y_test, scaler