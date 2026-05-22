import json
import joblib
import pandas as pd
import streamlit as st

from telco_churn.config import MODEL_PATH, FEATURE_COLS_PATH, BEST_THRESHOLD
from telco_churn.data.preprocess import preprocess

st.set_page_config(page_title="Churn Predictor", page_icon="📡", layout="wide")


@st.cache_resource
def load_model():
    """Load model and expected feature columns once, cache for all users."""
    if not MODEL_PATH.exists():
        return None, None
    model = joblib.load(MODEL_PATH)
    feature_cols = json.load(open(FEATURE_COLS_PATH))
    return model, feature_cols


model, feature_cols = load_model()

st.title("Telco Customer Churn Predictor")
st.caption(f"LightGBM · threshold = {BEST_THRESHOLD:.3f} · primary metric: recall")

if model is None:
    st.error("Model not found. Run `python scripts/run_pipeline.py` first.")
    st.stop()

# ── Sidebar: customer inputs ───────────────────────────────────────────────
st.sidebar.header("Customer details")

with st.sidebar:
    st.subheader("Personal")
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior citizen", ["No", "Yes"])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])

    st.subheader("Account")
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless billing", ["Yes", "No"])
    payment = st.selectbox("Payment method", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)",
    ])
    monthly = st.number_input("Monthly charges ($)", 0.0, 200.0, 65.0, step=0.5)
    total = st.number_input("Total charges ($)", 0.0, 10000.0, float(tenure * 65), step=1.0)

    st.subheader("Services")
    phone = st.selectbox("Phone service", ["Yes", "No"])
    no_phone = phone == "No"
    multi = st.selectbox(
        "Multiple lines",
        ["No phone service", "No", "Yes"],
        disabled=no_phone,
        index=0 if no_phone else 1,
    )

    internet = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
    no_internet = internet == "No"
    no_svc = "No internet service"

    online_sec = st.selectbox("Online security", [no_svc, "No", "Yes"], disabled=no_internet)
    online_bkp = st.selectbox("Online backup", [no_svc, "No", "Yes"], disabled=no_internet)
    device_prot = st.selectbox("Device protection", [no_svc, "No", "Yes"], disabled=no_internet)
    tech_sup = st.selectbox("Tech support", [no_svc, "No", "Yes"], disabled=no_internet)
    streaming_tv = st.selectbox("Streaming TV", [no_svc, "No", "Yes"], disabled=no_internet)
    streaming_mv = st.selectbox("Streaming movies", [no_svc, "No", "Yes"], disabled=no_internet)

    predict_btn = st.button("Predict churn risk", type="primary", use_container_width=True)

# ── Main area: result ──────────────────────────────────────────────────────
if predict_btn:
    raw = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": 1 if senior == "Yes" else 0,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multi,
        "InternetService": internet,
        "OnlineSecurity": online_sec,
        "OnlineBackup": online_bkp,
        "DeviceProtection": device_prot,
        "TechSupport": tech_sup,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_mv,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": str(total),  # raw data has this as string
        "Churn": "No",               # placeholder — dropped by preprocess
    }])

    processed = preprocess(raw)

    # Align to training columns: add any missing OHE columns as 0,
    # drop any extras (e.g. the placeholder Churn column).
    X = processed.reindex(columns=feature_cols, fill_value=0)

    proba = model.predict_proba(X)[0, 1]
    churns = proba >= BEST_THRESHOLD

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Churn probability", f"{proba:.1%}")

    with col2:
        if proba >= 0.65:
            st.error("High risk — recommend immediate outreach")
        elif proba >= 0.35:
            st.warning("Medium risk — monitor closely")
        else:
            st.success("Low risk — customer looks stable")

    st.divider()
    st.caption("**How to read this:** the model flags a customer as likely to churn "
               f"when their probability exceeds {BEST_THRESHOLD:.1%}. "
               "This threshold was chosen to maximise recall — we'd rather act on a "
               "false alarm than miss a real churner.")
else:
    st.info("Fill in the customer details in the sidebar and click **Predict churn risk**.")
