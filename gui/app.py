import streamlit as st
import requests
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Telco Churn Predictor", page_icon="🔮", layout="centered")

# --- API SETUP ---
# For now, it defaults to a placeholder. You will update this Env Var in Hugging Face.
API_URL = os.getenv("API_URL", "https://mlops-final-project-p939.onrender.com/predict")

# --- HEADER ---
st.title("🔮 Telco Customer Churn Prediction")
st.markdown(f"""
This interface is connected to a live MLOps pipeline running on **Render**.
Enter customer details below to generate a real-time prediction.
* **Backend:** FastAPI + XGBoost
* **Endpoint:** `{API_URL}`
""")

# --- INPUT FORM ---
# We use a form so the request is only sent when the user clicks "Predict"
with st.form("churn_form"):
    
    # SECTION 1: Demographics
    st.subheader("👤 Customer Info")
    col1, col2, col3 = st.columns(3)
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
    with col2:
        senior = st.selectbox("Senior Citizen", ["Yes", "No"])
    with col3:
        partner = st.selectbox("Partner", ["Yes", "No"])
        
    col4, col5 = st.columns(2)
    with col4:
        dependents = st.selectbox("Dependents", ["Yes", "No"])
    with col5:
        tenure = st.number_input("Tenure (Months)", min_value=0, max_value=120, value=12)

    st.divider()

    # SECTION 2: Services
    st.subheader("📡 Services")
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    
    with col_s2:
        online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])

    # Expandable section for less common features to keep UI clean
    with st.expander("Additional Tech Features"):
        tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

    st.divider()

    # SECTION 3: Billing
    st.subheader("💰 Contract & Payment")
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment_method = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
        ])
    with col_p2:
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=50.0, step=0.1)
        total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=500.0, step=1.0)

    # SUBMIT BUTTON
    submit_button = st.form_submit_button("🚀 Predict Churn Status")

# --- PREDICTION LOGIC ---
if submit_button:
    # 1. Prepare Payload (Must match Pydantic Model in api/app.py exactly)
    payload = {
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": int(tenure),
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment_method,
        "MonthlyCharges": float(monthly_charges),
        "TotalCharges": float(total_charges)
    }

    # 2. Send Request
    try:
        with st.spinner("Analyzing customer data..."):
            response = requests.post(API_URL, json=payload)
        
        # 3. Handle Response
        if response.status_code == 200:
            result = response.json()
            churn_prob = result["churn_probability"]
            prediction = result["churn_prediction"]
            
            # 4. Display Results
            st.success("Analysis Complete!")
            
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            with metric_col1:
                st.metric("Prediction", "Churn" if prediction == 1 else "Retention")
            with metric_col2:
                st.metric("Probability", f"{churn_prob:.2%}")
            with metric_col3:
                st.metric("Threshold Used", f"{result.get('threshold_used', 0.5):.2f}")

            # Visual Feedback
            st.progress(churn_prob)
            if prediction == 1:
                st.error("⚠️ HIGH RISK: This customer is likely to churn.")
            else:
                st.balloons()
                st.success("✅ LOW RISK: This customer is likely to stay.")
        
        else:
            st.error(f"❌ API Error {response.status_code}: {response.text}")

    except Exception as e:
        st.error(f"🔌 Connection Error: {e}")
        st.info("Hint: Make sure the 'API_URL' environment variable is correct in Hugging Face settings.")