"""
Dashboard page — model-level insights for the sales team.

Shows:
1. Global feature importance (which features matter most overall)
2. Full PDP curves for key features (how each feature affects churn on average)

No customer-specific data here — this is about understanding the model's
general behaviour so the team can prioritise outreach strategies.
"""
import sys
from pathlib import Path

# Make app/ importable so we can share utils with the Churn Predictor page
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import streamlit as st

from telco_churn.config import MULTI_CLASS_COLS
from utils import load_model, compute_pdp, PDP_FEATURES

st.set_page_config(page_title="Churn Dashboard", page_icon="📊", layout="wide")

model, feature_cols = load_model()
pdp_data = compute_pdp()

st.title("Churn Insights Dashboard")
st.caption(
    "Model-level view of what drives churn predictions. "
    "Use this to understand which levers matter most before diving into individual customers."
)

if model is None:
    st.error("Model not found. Run `python scripts/run_pipeline.py` first.")
    st.stop()


def _collapsed_importance(model, feature_cols: list) -> pd.DataFrame:
    """
    Combine OHE dummy-column importances back to their original feature name.
    e.g. Contract_One year + Contract_Two year → Contract
    """
    importance = {}
    for feat, imp in zip(feature_cols, model.feature_importances_):
        orig = next((o for o in MULTI_CLASS_COLS if feat.startswith(o + "_")), None)
        key  = orig if orig else feat
        importance[key] = importance.get(key, 0) + imp

    df = (
        pd.DataFrame({"feature": list(importance), "importance": list(importance.values())})
        .sort_values("importance", ascending=False)
    )
    # Normalise to 0-100 for readability
    df["importance"] = 100 * df["importance"] / df["importance"].sum()
    return df


# ── Section 1: Feature Importance ─────────────────────────────────────────────
st.subheader("Which features matter most?")
st.caption(
    "Importance = how often a feature is used to split decision trees in the model, "
    "summed across all trees and normalised to 100. Higher = the model relies on it more."
)

imp_df = _collapsed_importance(model, feature_cols).head(10)

fig, ax = plt.subplots(figsize=(8, 4))
bars = ax.barh(
    imp_df["feature"].iloc[::-1],
    imp_df["importance"].iloc[::-1],
    color="#2171b5", edgecolor="white", linewidth=0.4,
)
ax.set_xlabel("Relative importance (normalised, %)")
ax.set_title("Top 10 features by model importance")
ax.spines[["top", "right"]].set_visible(False)
for bar, val in zip(bars, imp_df["importance"].iloc[::-1]):
    ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}", va="center", fontsize=8)
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

st.divider()

# ── Section 2: PDP Charts ──────────────────────────────────────────────────────
st.subheader("How does each feature affect churn probability?")
st.caption(
    "Each chart holds all other features at a typical customer profile "
    "(29 months tenure, DSL internet, month-to-month contract, $65/month) "
    "and varies one feature across its full range. "
    "This shows the model's average sensitivity to each feature in isolation."
)


def _pdp_full_chart(feat: str, ax):
    cfg  = pdp_data[feat]
    prob = cfg["proba"]

    if cfg["type"] == "numeric":
        grid = cfg["grid"]
        ax.fill_between(grid, prob, alpha=0.12, color="#2171b5")
        ax.plot(grid, prob, color="#2171b5", linewidth=2)

        # Shade risk zones
        ax.axhspan(0,    0.2,  alpha=0.04, color="green")
        ax.axhspan(0.2,  0.5,  alpha=0.04, color="orange")
        ax.axhspan(0.5,  1.0,  alpha=0.04, color="red")

        ax.set_xlabel(cfg["label"], fontsize=10)

    else:  # categorical
        grid = cfg["grid"]
        ax.bar(range(len(grid)), prob, color="#2171b5", edgecolor="white", linewidth=0.5)
        ax.set_xticks(range(len(grid)))
        ax.set_xticklabels(grid, fontsize=9)

        for i, p in enumerate(prob):
            ax.text(i, p + 0.01, f"{p:.0%}", ha="center", fontsize=9, fontweight="bold")

        ax.set_xlabel(cfg["label"], fontsize=10)

    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.set_ylabel("Predicted churn probability", fontsize=9)
    ax.set_title(cfg["label"], fontsize=11, fontweight="bold", pad=6)
    ax.spines[["top", "right"]].set_visible(False)


feats = list(PDP_FEATURES.keys())
col_pairs = [st.columns(2), st.columns(2)]

for i, feat in enumerate(feats):
    row, col_idx = divmod(i, 2)
    with col_pairs[row][col_idx]:
        fig, ax = plt.subplots(figsize=(6, 4))
        _pdp_full_chart(feat, ax)
        fig.tight_layout(pad=1.5)
        st.pyplot(fig)
        plt.close(fig)
