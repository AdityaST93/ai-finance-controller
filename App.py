"""
App.py
-------
AI Finance Controller — Streamlit demo app.

Features:
  - Upload (or use sample) ledger + bank statement CSVs
  - AI-assisted reconciliation between the two
  - Anomaly / duplicate / outlier detection
  - Plain-language recommendations
  - Interactive dashboard (spend by vendor/category, trends)

Run locally:
    streamlit run App.py
"""

import os
import pandas as pd
import plotly.express as px
import streamlit as st

from reco_engine import (
    reconcile,
    detect_anomalies,
    vendor_spend_summary,
    generate_recommendations,
)
from generate_sample_data import generate_ledger, generate_bank_statement

st.set_page_config(
    page_title="AI Finance Controller",
    page_icon="💰",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Sidebar — data source
# ---------------------------------------------------------------------------

st.sidebar.title("💰 AI Finance Controller")
st.sidebar.caption("Automated reconciliation & spend intelligence")

data_mode = st.sidebar.radio(
    "Data source",
    ["Use sample data", "Upload my own CSVs"],
    index=0,
)

if data_mode == "Upload my own CSVs":
    ledger_file = st.sidebar.file_uploader("Ledger CSV", type="csv")
    bank_file = st.sidebar.file_uploader("Bank statement CSV", type="csv")
    if ledger_file and bank_file:
        ledger_df = pd.read_csv(ledger_file)
        bank_df = pd.read_csv(bank_file)
    else:
        st.sidebar.info("Upload both files to continue, or switch to sample data.")
        st.stop()
else:
    # Prefer bundled CSVs if present, else generate on the fly
    if os.path.exists("ledger.csv") and os.path.exists("bank_statement.csv"):
        ledger_df = pd.read_csv("ledger.csv")
        bank_df = pd.read_csv("bank_statement.csv")
    else:
        ledger_df = generate_ledger()
        bank_df = generate_bank_statement(ledger_df)

st.sidebar.markdown("---")
st.sidebar.metric("Ledger entries", len(ledger_df))
st.sidebar.metric("Bank entries", len(bank_df))

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("AI Finance Controller")
st.caption(
    "An AI-assisted co-pilot for reconciliation, anomaly detection, and spend visibility — "
    "built for finance teams who are tired of manually matching spreadsheets."
)

with st.spinner("Running reconciliation engine..."):
    matched_df, unexplained_bank = reconcile(ledger_df, bank_df)
    anomalies_df = detect_anomalies(ledger_df)
    vendor_summary = vendor_spend_summary(ledger_df)
    recommendations = generate_recommendations(matched_df, unexplained_bank, anomalies_df)

# ---------------------------------------------------------------------------
# Top KPIs
# ---------------------------------------------------------------------------

col1, col2, col3, col4 = st.columns(4)

total_spend = ledger_df["amount"].sum()
matched_count = (matched_df["status"] == "Matched").sum()
match_rate = matched_count / len(matched_df) * 100 if len(matched_df) else 0
issues_count = (
    (matched_df["status"] != "Matched").sum() + len(unexplained_bank)
)

col1.metric("Total Ledger Spend", f"₹{total_spend:,.0f}")
col2.metric("Reconciliation Match Rate", f"{match_rate:.1f}%")
col3.metric("Flagged Issues", int(issues_count))
col4.metric("Vendors Tracked", ledger_df["vendor"].nunique())

st.markdown("---")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab1, tab2, tab3, tab4 = st.tabs(
    ["🤖 AI Recommendations", "🔗 Reconciliation", "🚨 Anomalies", "📊 Spend Dashboard"]
)

with tab1:
    st.subheader("AI-generated insights")
    for tip in recommendations:
        st.info(tip)

with tab2:
    st.subheader("Ledger vs. Bank Statement matching")

    status_filter = st.multiselect(
        "Filter by status",
        options=matched_df["status"].unique().tolist(),
        default=matched_df["status"].unique().tolist(),
    )
    filtered = matched_df[matched_df["status"].isin(status_filter)]
    st.dataframe(
        filtered[["date", "vendor", "category", "amount", "bank_amount", "amount_diff", "status"]],
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("**Bank transactions not found in ledger**")
    st.dataframe(unexplained_bank, use_container_width=True, hide_index=True)

with tab3:
    st.subheader("Outliers & possible duplicate payments")

    outliers = anomalies_df[anomalies_df["is_outlier"]]
    dupes = anomalies_df[anomalies_df["is_possible_duplicate"]]

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**Statistical outliers** ({len(outliers)})")
        st.dataframe(
            outliers[["date", "vendor", "amount", "z_score"]].sort_values("z_score", ascending=False),
            use_container_width=True,
            hide_index=True,
        )
    with c2:
        st.markdown(f"**Possible duplicate payments** ({len(dupes)})")
        st.dataframe(
            dupes[["date", "vendor", "amount", "recorded_by"]],
            use_container_width=True,
            hide_index=True,
        )

with tab4:
    st.subheader("Spend visibility")

    c1, c2 = st.columns(2)
    with c1:
        cat_spend = ledger_df.groupby("category")["amount"].sum().reset_index()
        fig1 = px.pie(cat_spend, names="category", values="amount", title="Spend by Category", hole=0.4)
        st.plotly_chart(fig1, use_container_width=True)
    with c2:
        top_vendors = vendor_summary.head(10)
        fig2 = px.bar(
            top_vendors, x="total_spend", y="vendor", orientation="h",
            title="Top 10 Vendors by Spend", labels={"total_spend": "Total Spend (₹)", "vendor": ""},
        )
        fig2.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig2, use_container_width=True)

    ledger_df["date"] = pd.to_datetime(ledger_df["date"])
    daily = ledger_df.groupby(ledger_df["date"].dt.to_period("W").astype(str))["amount"].sum().reset_index()
    daily.columns = ["week", "amount"]
    fig3 = px.line(daily, x="week", y="amount", markers=True, title="Weekly Spend Trend")
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("**Full vendor summary**")
    st.dataframe(vendor_summary, use_container_width=True, hide_index=True)

st.markdown("---")
st.caption("Built with Streamlit · Demo data is synthetic and for illustration only.")
