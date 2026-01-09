import pandas as pd
import requests
import time
import sys
from sklearn.model_selection import train_test_split

# CONFIG
# API_URL = "https://mlops-final-project-p939.onrender.com/predict"
API_URL = "http://localhost:8000/predict"  # Local
DATA_PATH = "archive/WA_Fn-UseC_-Telco-Customer-Churn.csv"
SPEED = 0.5  # Seconds between requests (Lower = Faster)

def load_raw_test_set():
    print("⏳ Loading Raw Data...")
    df = pd.read_csv(DATA_PATH)
    
    # 1. Clean TotalCharges (coerce spaces to NaN then 0)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
    
    # 2. Strict Train/Test Split (Must match your training logic!)
    # We drop 'customerID' and 'Churn' as they aren't features
    X = df.drop(columns=['customerID', 'Churn'])
    y = df['Churn']
    
    # Use the same seed as training to ensure this IS the test set
    X_train, X_test, _, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"✅ Loaded Test Set: {len(X_test)} samples.")
    return X_test

def adapt_row_to_api(row):
    """
    Converts a Raw CSV row to the exact format expected by the API Pydantic model.
    """
    data = row.to_dict()
    
    # FIX: SeniorCitizen in CSV is 0/1, but API expects "Yes"/"No"
    if data['SeniorCitizen'] == 1:
        data['SeniorCitizen'] = "Yes"
    else:
        data['SeniorCitizen'] = "No"
        
    # Ensure numerics are Python floats/ints, not numpy types
    data['tenure'] = int(data['tenure'])
    data['MonthlyCharges'] = float(data['MonthlyCharges'])
    data['TotalCharges'] = float(data['TotalCharges'])
    
    return data

def main():
    X_test = load_raw_test_set()
    
    print(f"🚀 Streaming {len(X_test)} test samples to {API_URL}...")
    print("Press Ctrl+C to stop.\n")
    
    counter = 0
    try:
        # Loop through the test set
        for index, row in X_test.iterrows():
            payload = adapt_row_to_api(row)
            
            # Send Request
            try:
                response = requests.post(API_URL, json=payload)
                
                if response.status_code == 200:
                    result = response.json()
                    prob = result['churn_probability']
                    pred = "🔴 Churn" if result['churn_prediction'] == 1 else "🟢 Stay"
                    
                    print(f"[{counter}] {pred} (Prob: {prob:.4f}) | Customer Tenure: {payload['tenure']}")
                else:
                    print(f"❌ Error {response.status_code}: {response.text}")
                    
            except requests.exceptions.ConnectionError:
                print("⚠️ API Connection Failed. Is uvicorn running?")
                time.sleep(1)
            
            counter += 1
            time.sleep(SPEED)
            
    except KeyboardInterrupt:
        print("\n🛑 Stream stopped.")

if __name__ == "__main__":
    main()
