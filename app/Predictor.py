import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from telco_churn.config import BEST_THRESHOLD, MULTI_CLASS_COLS
from telco_churn.data.preprocess import preprocess
from utils import load_model, predict_single, compute_pdp, PDP_FEATURES

st.set_page_config(page_title="Churn Predictor", page_icon="📡", layout="wide")

model, feature_cols = load_model()
pdp_data = compute_pdp()

st.title("Telco Customer Churn Predictor")
st.caption(
    f"LightGBM · threshold = {BEST_THRESHOLD:.3f} · "
    "primary metric: recall (we'd rather flag a false alarm than miss a real churner)"
)

if model is None:
    st.error("Model not found. Run `python scripts/run_pipeline.py` first.")
    st.stop()


def _build_counterfactuals(raw_values: dict, current_proba: float) -> list[dict]:
    base = {**raw_values}
    scenarios = []

    for alt in ["One year", "Two year"]:
        if alt != raw_values.get("Contract"):
            p = predict_single(model, feature_cols, {**base, "Contract": alt})
            scenarios.append({"change": f"Switch to {alt} contract", "new_proba": p})

    if raw_values.get("TechSupport") == "No" and raw_values.get("InternetService") != "No":
        p = predict_single(model, feature_cols, {**base, "TechSupport": "Yes"})
        scenarios.append({"change": "Add tech support", "new_proba": p})

    if raw_values.get("OnlineSecurity") == "No" and raw_values.get("InternetService") != "No":
        p = predict_single(model, feature_cols, {**base, "OnlineSecurity": "Yes"})
        scenarios.append({"change": "Add online security", "new_proba": p})

    if raw_values.get("PaymentMethod") == "Electronic check":
        p = predict_single(model, feature_cols, {**base, "PaymentMethod": "Bank transfer (automatic)"})
        scenarios.append({"change": "Switch to bank transfer", "new_proba": p})

    scenarios = [s for s in scenarios if s["new_proba"] < current_proba - 0.01]
    scenarios.sort(key=lambda s: s["new_proba"])
    return scenarios


def _pdp_mini_chart(feat: str, customer_val, ax):
    """
    Draw a single PDP curve on ax and mark where the customer falls.
    For numeric features: line chart + red dot + vertical guide.
    For categorical: bar chart with the customer's bar highlighted red.
    """
    cfg = pdp_data[feat]
    proba = cfg["proba"]

    if cfg["type"] == "numeric":
        grid = cfg["grid"]
        ax.fill_between(grid, proba, alpha=0.08, color="#2171b5")
        ax.plot(grid, proba, color="#2171b5", linewidth=1.8)

        customer_y = float(np.interp(customer_val, grid, proba))
        ax.axvline(customer_val, color="#888", linestyle="--", linewidth=1, alpha=0.7)
        ax.scatter([customer_val], [customer_y], color="#d62728", s=60, zorder=5)
        ax.annotate(
            f"{customer_y:.0%}",
            xy=(customer_val, customer_y),
            xytext=(6, 4), textcoords="offset points",
            fontsize=8, color="#d62728", fontweight="bold",
        )

    else:  # categorical
        grid = cfg["grid"]
        colors = ["#d62728" if c == customer_val else "#2171b5" for c in grid]
        ax.bar(range(len(grid)), proba, color=colors, edgecolor="white", linewidth=0.5)
        ax.set_xticks(range(len(grid)))
        ax.set_xticklabels(grid, fontsize=7, rotation=12, ha="right")

        customer_y = proba[grid.index(customer_val)] if customer_val in grid else None
        if customer_y is not None:
            ax.annotate(
                f"{customer_y:.0%}",
                xy=(grid.index(customer_val), customer_y),
                xytext=(0, 4), textcoords="offset points",
                fontsize=8, color="#d62728", fontweight="bold", ha="center",
            )

    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.set_title(cfg["label"], fontsize=9, fontweight="bold", pad=4)
    ax.tick_params(axis="both", labelsize=7)
    ax.spines[["top", "right"]].set_visible(False)


# ── Section 1: Personal ────────────────────────────────────────────────────────
st.subheader("Personal")
c1, c2, c3, c4 = st.columns(4)
gender     = c1.selectbox("Gender", ["Male", "Female"])
senior     = c2.selectbox("Senior citizen", ["No", "Yes"])
partner    = c3.selectbox("Partner", ["Yes", "No"])
dependents = c4.selectbox("Dependents", ["Yes", "No"])

st.divider()

# ── Section 2: Account ─────────────────────────────────────────────────────────
st.subheader("Account")
c1, c2, c3 = st.columns(3)
with c1:
    tenure   = st.slider("Tenure (months)", 0, 72, 12)
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
with c2:
    monthly = st.number_input("Monthly charges ($)", 0.0, 200.0, 65.0, step=0.5)
    total   = st.number_input("Total charges ($)", 0.0, 10000.0, float(tenure * 65), step=1.0)
with c3:
    paperless = st.selectbox("Paperless billing", ["Yes", "No"])
    payment   = st.selectbox("Payment method", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)",
    ])

st.divider()

# ── Section 3: Services ────────────────────────────────────────────────────────
st.subheader("Services")
no_svc = "No internet service"

c1, c2, c3 = st.columns(3)
with c1:
    phone    = st.selectbox("Phone service", ["Yes", "No"])
    no_phone = phone == "No"
    multi    = st.selectbox(
        "Multiple lines",
        ["No phone service", "No", "Yes"],
        disabled=no_phone,
        index=0 if no_phone else 1,
    )
    internet    = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
    no_internet = internet == "No"
with c2:
    online_sec  = st.selectbox("Online security",   [no_svc, "No", "Yes"], disabled=no_internet)
    online_bkp  = st.selectbox("Online backup",     [no_svc, "No", "Yes"], disabled=no_internet)
    device_prot = st.selectbox("Device protection", [no_svc, "No", "Yes"], disabled=no_internet)
with c3:
    tech_sup     = st.selectbox("Tech support",     [no_svc, "No", "Yes"], disabled=no_internet)
    streaming_tv = st.selectbox("Streaming TV",     [no_svc, "No", "Yes"], disabled=no_internet)
    streaming_mv = st.selectbox("Streaming movies", [no_svc, "No", "Yes"], disabled=no_internet)

st.divider()

# ── Predict button ─────────────────────────────────────────────────────────────
_, btn_col, _ = st.columns([3, 1, 3])
predict_btn = btn_col.button("Predict Churn Risk", type="primary", use_container_width=True)

# ── Results ────────────────────────────────────────────────────────────────────
if predict_btn:
    raw_values = {
        "gender":           gender,
        "SeniorCitizen":    1 if senior == "Yes" else 0,
        "Partner":          partner,
        "Dependents":       dependents,
        "tenure":           tenure,
        "PhoneService":     phone,
        "MultipleLines":    multi,
        "InternetService":  internet,
        "OnlineSecurity":   online_sec,
        "OnlineBackup":     online_bkp,
        "DeviceProtection": device_prot,
        "TechSupport":      tech_sup,
        "StreamingTV":      streaming_tv,
        "StreamingMovies":  streaming_mv,
        "Contract":         contract,
        "PaperlessBilling": paperless,
        "PaymentMethod":    payment,
        "MonthlyCharges":   monthly,
        "TotalCharges":     str(total),
        "Churn":            "No",
    }

    proba  = predict_single(model, feature_cols, raw_values)
    churns = proba >= BEST_THRESHOLD

    st.divider()
    st.subheader("Prediction Results")

    m1, m2, m3 = st.columns(3)
    m1.metric("Churn probability",  f"{proba:.1%}")
    m2.metric("Decision threshold", f"{BEST_THRESHOLD:.1%}")
    m3.metric("Predicted outcome",  "Will churn" if churns else "Will stay")

    if proba >= 0.65:
        st.error("High risk — recommend immediate outreach")
    elif churns:
        st.warning("Medium risk — monitor closely")
    else:
        st.success("Low risk — customer looks stable")

    # ── PDP position charts ────────────────────────────────────────────────────
    st.divider()
    st.subheader("Where does this customer sit?")
    st.caption(
        "Each chart shows the average churn probability across all customers as a feature varies "
        "(blue curve / bars). The red marker shows where **this customer** lands. "
        "Charts are based on a typical baseline customer — see the Dashboard page for full context."
    )

    customer_vals = {
        "tenure":          tenure,
        "MonthlyCharges":  monthly,
        "Contract":        contract,
        "InternetService": internet,
    }

    feats = list(PDP_FEATURES.keys())
    col_pairs = [st.columns(2), st.columns(2)]

    for i, feat in enumerate(feats):
        row, col_idx = divmod(i, 2)
        with col_pairs[row][col_idx]:
            fig, ax = plt.subplots(figsize=(4.5, 3))
            _pdp_mini_chart(feat, customer_vals[feat], ax)
            fig.tight_layout(pad=1.2)
            st.pyplot(fig)
            plt.close(fig)

    # ── Counterfactuals ────────────────────────────────────────────────────────
    st.divider()
    st.subheader("What would reduce this customer's risk?")
    st.caption("Each scenario re-runs the model with one change. Only improvements > 1% are shown.")

    scenarios = _build_counterfactuals(raw_values, proba)

    if not scenarios:
        st.success(
            "No single change meaningfully reduces this customer's risk — "
            "the risk profile is driven by multiple compounding factors."
        )
    else:
        cols = st.columns(min(len(scenarios), 3))
        for col, s in zip(cols, scenarios):
            col.metric(
                label=s["change"],
                value=f"{s['new_proba']:.1%}",
                delta=f"−{proba - s['new_proba']:.1%} reduction",
                delta_color="inverse",
            )

else:
    st.info("Fill in the customer details above and click **Predict Churn Risk** to see results.")
