import streamlit as st
import pandas as pd
import joblib


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Bank Marketing Prediction",
    page_icon="🏦",
    layout="wide"
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load("model.pkl")


model = load_model()


# ============================================================
# TITLE
# ============================================================

st.title("🏦 Bank Marketing Campaign Prediction")

st.write(
    "This application predicts whether a customer is likely "
    "to subscribe to a bank term deposit."
)

st.divider()


# ============================================================
# INPUT SECTION
# ============================================================

st.header("Customer Information")


col1, col2, col3 = st.columns(3)


with col1:

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=30
    )

    job = st.selectbox(
        "Job",
        [
            "admin.",
            "blue-collar",
            "entrepreneur",
            "housemaid",
            "management",
            "retired",
            "self-employed",
            "services",
            "student",
            "technician",
            "unemployed",
            "unknown"
        ]
    )

    marital = st.selectbox(
        "Marital Status",
        [
            "married",
            "single",
            "divorced"
        ]
    )

    education = st.selectbox(
        "Education",
        [
            "primary",
            "secondary",
            "tertiary",
            "unknown"
        ]
    )

    default = st.selectbox(
        "Has Credit Default?",
        [
            "no",
            "yes",
            "unknown"
        ]
    )


with col2:

    balance = st.number_input(
        "Account Balance",
        value=0
    )

    housing = st.selectbox(
        "Housing Loan?",
        [
            "yes",
            "no"
        ]
    )

    loan = st.selectbox(
        "Personal Loan?",
        [
            "yes",
            "no"
        ]
    )

    contact = st.selectbox(
        "Contact Type",
        [
            "cellular",
            "telephone",
            "unknown"
        ]
    )

    day = st.number_input(
        "Last Contact Day",
        min_value=1,
        max_value=31,
        value=15
    )


with col3:

    month = st.selectbox(
        "Last Contact Month",
        [
            "jan",
            "feb",
            "mar",
            "apr",
            "may",
            "jun",
            "jul",
            "aug",
            "sep",
            "oct",
            "nov",
            "dec"
        ]
    )

    duration = st.number_input(
        "Call Duration (seconds)",
        min_value=0,
        value=100
    )

    campaign = st.number_input(
        "Number of Contacts During Campaign",
        min_value=1,
        value=1
    )

    pdays = st.number_input(
        "Days Since Previous Contact",
        value=-1
    )

    previous = st.number_input(
        "Number of Previous Contacts",
        min_value=0,
        value=0
    )

    poutcome = st.selectbox(
        "Previous Campaign Outcome",
        [
            "unknown",
            "failure",
            "other",
            "success"
        ]
    )


# ============================================================
# PREDICTION BUTTON
# ============================================================

st.divider()

if st.button(
    "🔮 Predict",
    use_container_width=True
):

    # Create input dataframe

    input_data = pd.DataFrame({
        "age": [age],
        "job": [job],
        "marital": [marital],
        "education": [education],
        "default": [default],
        "balance": [balance],
        "housing": [housing],
        "loan": [loan],
        "contact": [contact],
        "day": [day],
        "month": [month],
        "duration": [duration],
        "campaign": [campaign],
        "pdays": [pdays],
        "previous": [previous],
        "poutcome": [poutcome]
    })


    # Make prediction

    prediction = model.predict(input_data)[0]


    # Probability

    probability = model.predict_proba(input_data)[0]


    # ========================================================
    # DISPLAY RESULT
    # ========================================================

    if prediction == 1:

        st.success(
            "✅ Prediction: Customer is likely to subscribe."
        )

        st.write(
            f"Probability of subscription: "
            f"{probability[1] * 100:.2f}%"
        )

    else:

        st.warning(
            "❌ Prediction: Customer is unlikely to subscribe."
        )

        st.write(
            f"Probability of subscription: "
            f"{probability[0] * 100:.2f}%"
        )


    # Show input

    st.subheader("Customer Information")

    st.dataframe(
        input_data,
        use_container_width=True
    )