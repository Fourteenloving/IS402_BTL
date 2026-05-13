import io
import os

import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components

API_URL = os.getenv("API_URL", "http://api:8000/predict")
BATCH_URL = os.getenv("BATCH_URL", "http://api:8000/batch")
DRIFT_URL = os.getenv("DRIFT_URL", "http://api:8000/monitor/drift?format=html")
DRIFT_JSON_URL = os.getenv("DRIFT_JSON_URL", "http://api:8000/monitor/drift?format=json")
REQUIRED_COLUMNS = [
    "Age",
    "Gender",
    "Tenure",
    "Usage Frequency",
    "Support Calls",
    "Payment Delay",
    "Subscription Type",
    "Contract Length",
    "Total Spend",
    "Last Interaction",
]

st.set_page_config(page_title="Customer Churn Prediction", layout="wide")


def _normalize_batch_frame(frame: pd.DataFrame) -> pd.DataFrame:
    renamed = frame.rename(
        columns={
            "Usage_Frequency": "Usage Frequency",
            "Support_Calls": "Support Calls",
            "Payment_Delay": "Payment Delay",
            "Subscription_Type": "Subscription Type",
            "Contract_Length": "Contract Length",
            "Total_Spend": "Total Spend",
            "Last_Interaction": "Last Interaction",
        }
    )
    missing = [column for column in REQUIRED_COLUMNS if column not in renamed.columns]
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing)}")
    return renamed[REQUIRED_COLUMNS].copy()


def _post_json(url: str, payload):
    response = requests.post(url, json=payload, timeout=30)
    response.raise_for_status()
    return response.json()


st.title("Customer Churn Prediction")
single_tab, batch_tab, drift_tab = st.tabs(["Single Prediction", "Batch Prediction", "Data Drift"])

with single_tab:
    left, right = st.columns(2)
    with left:
        age = st.slider("Age", min_value=18, max_value=100, value=30)
        gender = st.radio("Gender", options=["Male", "Female"], horizontal=True)
        tenure = st.slider("Tenure", min_value=0, max_value=72, value=12)
        usage_frequency = st.slider("Usage Frequency", min_value=0, max_value=30, value=10)
        support_calls = st.slider("Support Calls", min_value=0, max_value=10, value=2)
    with right:
        payment_delay = st.slider("Payment Delay", min_value=0, max_value=30, value=5)
        subscription_type = st.selectbox("Subscription Type", options=["Basic", "Standard", "Premium"], index=1)
        contract_length = st.selectbox("Contract Length", options=["Monthly", "Quarterly", "Annual"])
        total_spend = st.number_input("Total Spend", min_value=0.0, value=500.0, step=10.0)
        last_interaction = st.slider("Last Interaction", min_value=0, max_value=30, value=15)

    if st.button("Predict Churn", use_container_width=True):
        payload = {
            "Age": age,
            "Gender": gender,
            "Tenure": tenure,
            "Usage Frequency": usage_frequency,
            "Support Calls": support_calls,
            "Payment Delay": payment_delay,
            "Subscription Type": subscription_type,
            "Contract Length": contract_length,
            "Total Spend": total_spend,
            "Last Interaction": last_interaction,
        }
        try:
            result = _post_json(API_URL, payload)
            level = "error" if result["prediction"] == 1 else "success"
            getattr(st, level)(f"Prediction: {result['status']}")
        except requests.RequestException as exc:
            st.error(f"Request failed: {exc}")

with batch_tab:
    uploaded_file = st.file_uploader("Upload CSV", type="csv")
    if uploaded_file:
        try:
            frame = _normalize_batch_frame(pd.read_csv(io.BytesIO(uploaded_file.getvalue())))
            st.dataframe(frame.head(20), use_container_width=True)
            if st.button("Run Batch Prediction", use_container_width=True):
                result = _post_json(BATCH_URL, frame.to_dict(orient="records"))
                predictions = pd.DataFrame(result["predictions"])
                output = pd.concat([frame.reset_index(drop=True), predictions], axis=1)
                st.dataframe(output, use_container_width=True)
                churn_count = int(predictions["prediction"].sum())
                st.info(f"Rows: {result['total']} | Churn: {churn_count} | No Churn: {result['total'] - churn_count}")
        except Exception as exc:
            st.error(str(exc))

with drift_tab:
    left, right = st.columns([1, 3])
    with left:
        if st.button("Load Drift Summary", use_container_width=True):
            try:
                summary = requests.get(DRIFT_JSON_URL, timeout=60).json()
                st.json(summary)
            except requests.RequestException as exc:
                st.error(f"Request failed: {exc}")
        if st.button("Load Drift Report", use_container_width=True):
            try:
                st.session_state["drift_html"] = requests.get(DRIFT_URL, timeout=120).text
            except requests.RequestException as exc:
                st.error(f"Request failed: {exc}")
    with right:
        if "drift_html" in st.session_state:
            components.html(st.session_state["drift_html"], height=900, scrolling=True)
