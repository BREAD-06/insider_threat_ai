# app.py
# Streamlit dashboard for visualising CERT r4.2 insider threat detection results.

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import pandas as pd
import plotly.express as px

from pipeline import run_pipeline

st.set_page_config(page_title="Insider Threat AI", page_icon="🛡️", layout="wide")

st.title("🛡️ Insider Threat Detection Dashboard")
st.markdown(
    "Real-time analysis of **CERT r4.2** employee activity logs "
    "using an AI-powered multi-agent pipeline (Isolation Forest)."
)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")
    data_dir   = st.text_input("CERT data folder", value="data/cert_r4.2")
    model_path = st.text_input("Model path",        value="models/isolation_forest.pkl")
    run_btn    = st.button("▶  Run Pipeline", type="primary")

# ── Run ───────────────────────────────────────────────────────────────────────
if run_btn:
    with st.spinner("Running pipeline — this may take a minute on the full CERT dataset…"):
        try:
            result = run_pipeline(data_dir=data_dir, model_path=model_path)
        except Exception as e:
            st.error(f"Pipeline error: {e}")
            st.stop()

    df_v: pd.DataFrame = result["verified_df"]

    # ── KPI row ───────────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    c1.metric("Feature Rows (user×hour)", f"{result['total']:,}")
    c2.metric("Anomalies Detected",       f"{result['anomalies']:,}")
    c3.metric("Confirmed Threats",        f"{result['confirmed']:,}")

    st.divider()

    # ── Verified threats table ────────────────────────────────────────────────
    st.subheader("📋 Verified Threats")
    st.dataframe(df_v, use_container_width=True)

    st.divider()

    # ── Chart 1 : Activity by hour ────────────────────────────────────────────
    if "hour" in df_v.columns:
        st.subheader("🕐 Threat Activity by Hour of Day")
        fig = px.histogram(
            df_v, x="hour",
            color="confirmed_threat" if "confirmed_threat" in df_v.columns else None,
            barmode="overlay",
            nbins=24,
            labels={"hour": "Hour of Day", "confirmed_threat": "Confirmed Threat"},
            color_discrete_map={True: "#ef4444", False: "#6366f1"},
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Chart 2 : Anomaly score distribution ──────────────────────────────────
    if "anomaly_score" in df_v.columns:
        st.subheader("📊 Anomaly Score Distribution")
        fig2 = px.box(
            df_v, y="anomaly_score",
            color="confirmed_threat" if "confirmed_threat" in df_v.columns else None,
            points="outliers",
            color_discrete_map={True: "#ef4444", False: "#6366f1"},
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ── Chart 3 : USB activity ────────────────────────────────────────────────
    if "usb_connect" in df_v.columns and df_v["usb_connect"].sum() > 0:
        st.subheader("🔌 USB Connect Events per User (Flagged Records)")
        usb_df = (
            df_v.groupby("user")[["usb_connect", "usb_disconnect"]]
            .sum()
            .reset_index()
            .sort_values("usb_connect", ascending=False)
            .head(20)
        )
        fig3 = px.bar(
            usb_df, x="user", y=["usb_connect", "usb_disconnect"],
            barmode="group",
            labels={"user": "User", "value": "Events", "variable": "Type"},
            color_discrete_map={"usb_connect": "#f59e0b", "usb_disconnect": "#64748b"},
        )
        st.plotly_chart(fig3, use_container_width=True)

    # ── Chart 4 : Email volume per user ───────────────────────────────────────
    if "email_count" in df_v.columns and df_v["email_count"].sum() > 0:
        st.subheader("📧 Email Volume per User (Flagged Records)")
        email_df = (
            df_v.groupby("user")[["email_count", "email_size_total"]]
            .sum()
            .reset_index()
            .sort_values("email_count", ascending=False)
            .head(20)
        )
        fig4 = px.bar(
            email_df, x="user", y="email_count",
            color="email_size_total",
            color_continuous_scale="OrRd",
            labels={"user": "User", "email_count": "Emails Sent", "email_size_total": "Total Size (bytes)"},
        )
        st.plotly_chart(fig4, use_container_width=True)

    # ── Chart 5 : File & HTTP activity ───────────────────────────────────────
    if "file_count" in df_v.columns:
        st.subheader("📂 File & Web Activity per User (Flagged Records)")
        act_df = (
            df_v.groupby("user")[["file_count", "http_count"]]
            .sum()
            .reset_index()
            .sort_values("file_count", ascending=False)
            .head(20)
        )
        fig5 = px.bar(
            act_df, x="user", y=["file_count", "http_count"],
            barmode="group",
            labels={"user": "User", "value": "Count", "variable": "Activity"},
            color_discrete_map={"file_count": "#3b82f6", "http_count": "#8b5cf6"},
        )
        st.plotly_chart(fig5, use_container_width=True)

else:
    st.info(
        "Set the **CERT data folder** path in the sidebar "
        "(default: `data/cert_r4.2`), then click **▶ Run Pipeline** to begin."
    )
