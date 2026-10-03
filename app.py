# ============================================================
# BANK MARKETING PREDICTION — STREAMLIT APP (app.py)
# ============================================================

import os
import joblib
import pandas as pd
import streamlit as st
import urllib.request

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="Bank Term Deposit Prediction",
    page_icon="🏦",
    layout="wide"
)

# ------------------------------------------------------------
# 1. LOAD MODEL PIPELINE FROM GOOGLE DRIVE (DIRECT URLLIB DOWNLOAD)
# ------------------------------------------------------------
# ------------------------------------------------------------
# 1. LOAD MODEL PIPELINE FROM GOOGLE DRIVE (ROBUST GDOWN)
# ------------------------------------------------------------
FILE_ID = "1IeM5dID45LHFEwX3UgovjJeQkpnlOgWW"
MODEL_FILE = "model.pkl"

@st.cache_resource
def load_pipeline(file_id: str, output_path: str):
    """Downloads model.pkl safely using gdown with fuzzy matching."""
    if not os.path.exists(output_path) or os.path.getsize(output_path) < 1000:
        with st.spinner("Downloading trained pipeline model from Google Drive..."):
            import gdown
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, output_path, quiet=False, fuzzy=True)
            
    if not os.path.exists(output_path) or os.path.getsize(output_path) < 1000:
        raise ValueError("Downloaded file is too small or invalid. Please ensure Google Drive sharing permissions are set to 'Anyone with the link can view'.")
        
    pipeline = joblib.load(output_path)
    return pipeline

# Load model pipeline with error handling
try:
    pipeline = load_pipeline(FILE_ID, MODEL_FILE)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.info("Ensure your Google Drive file access is set to 'Anyone with the link can view'.")
    st.stop()
     


# ------------------------------------------------------------
# 2. USER INTERFACE & INPUT FORM
# ------------------------------------------------------------
st.title("🏦 Bank Term Deposit Prediction")
st.markdown("Enter customer details below to predict if they will subscribe to a bank term deposit.")

st.sidebar.header("Customer Input Form")

# Collect inputs matching feature columns from training dataset
age = st.sidebar.slider("Age", 18, 95, 35)
job = st.sidebar.selectbox(
    "Job Type",
    ["admin.", "blue-collar", "technician", "services", "management", 
     "retired", "entrepreneur", "self-employed", "housemaid", "unemployed", "student", "unknown"]
)
marital = st.sidebar.selectbox("Marital Status", ["married", "single", "divorced"])
education = st.sidebar.selectbox("Education Level", ["secondary", "tertiary", "primary", "unknown"])
default = st.sidebar.selectbox("Credit in Default?", ["no", "yes"])
balance = st.sidebar.number_input("Average Yearly Balance (€)", value=1000)
housing = st.sidebar.selectbox("Housing Loan?", ["no", "yes"])
loan = st.sidebar.selectbox("Personal Loan?", ["no", "yes"])
contact = st.sidebar.selectbox("Contact Communication Type", ["cellular", "telephone", "unknown"])
day = st.sidebar.slider("Last Contact Day of Month", 1, 31, 15)
month = st.sidebar.selectbox(
    "Last Contact Month", 
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
)
duration = st.sidebar.number_input("Last Contact Duration (seconds)", value=180)
campaign = st.sidebar.number_input("Number of Contacts during Campaign", min_value=1, value=1)
pdays = st.sidebar.number_input("Days since previous campaign contact (-1 = never)", value=-1)
previous = st.sidebar.number_input("Number of Contacts before Campaign", value=0)
poutcome = st.sidebar.selectbox("Outcome of Previous Marketing Campaign", ["unknown", "other", "failure", "success"])


# Construct DataFrame matching training raw feature names
input_dict = {
    "age": age,
    "job": job,
    "marital": marital,
    "education": education,
    "default": default,
    "balance": balance,
    "housing": housing,
    "loan": loan,
    "contact": contact,
    "day": day,
    "month": month,
    "duration": duration,
    "campaign": campaign,
    "pdays": pdays,
    "previous": previous,
    "poutcome": poutcome
}

input_df = pd.DataFrame([input_dict])

st.subheader("Customer Input Summary")
st.dataframe(input_df)


# ------------------------------------------------------------
# 3. PREDICTION & DISPLAY RESULTS
# ------------------------------------------------------------
if st.button("Predict Subscription", type="primary"):
    try:
        # Pipeline automatically handles categorical encoding and scaling
        prediction = pipeline.predict(input_df)[0]
        proba = pipeline.predict_proba(input_df)[0][1]

        st.markdown("---")
        st.subheader("Prediction Result")

        if prediction == 1 or str(prediction).lower() == "yes":
            st.success(f"🎉 **High Likelihood to Subscribe!** (Probability: {round(proba * 100, 2)}%)")
        else:
            st.warning(f"⚠️ **Unlikely to Subscribe.** (Probability: {round(proba * 100, 2)}%)")

    except Exception as err:
        st.error(f"Prediction failed: {err}")