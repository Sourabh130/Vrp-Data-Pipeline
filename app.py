"""
Production-grade Streamlit web interface for interactive VRP data cleaning,
preprocessing, exploratory analysis visualization, and split downloads.
"""

import json
import os
from pathlib import Path
import pandas as pd
import streamlit as st

from src.config import (
    CORR_THRESHOLD,
    IQR_MULTIPLIER,
    MISSING_COL_THRESHOLD,
    OUTPUTS_DIR,
    PLOTS_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    VALID_SCALERS,
    VALID_SPLIT_MODES,
    PipelineConfig,
)
from src.pipeline import run_pipeline

st.set_page_config(
    page_title="VRP EV Telemetry Preprocessing Pipeline",
    page_icon="⚡",
    layout="wide"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1e3a8a;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #4b5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">⚡ VRP Telemetry Data Preprocessing Pipeline</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Production-grade, dataset-agnostic cleaning & feature engineering for JEV Model Analysis</div>', unsafe_allow_html=True)

# Sidebar: Configuration
st.sidebar.header("🛠️ Pipeline Parameters")

# File Selection / Upload
data_source = st.sidebar.radio("Data Source", ["Select Existing Raw CSV", "Upload New CSV"])
input_file_path = None

if data_source == "Select Existing Raw CSV":
    raw_files = list(RAW_DATA_DIR.glob("*.csv")) if RAW_DATA_DIR.exists() else []
    if raw_files:
        chosen_file = st.sidebar.selectbox("Choose Raw CSV", [f.name for f in raw_files])
        input_file_path = RAW_DATA_DIR / chosen_file
    else:
        st.sidebar.warning("No CSV files found in data/raw/. A synthetic sample will be auto-generated.")
        input_file_path = RAW_DATA_DIR / "sample_vrp.csv"
else:
    uploaded_file = st.sidebar.file_uploader("Upload VRP / EV CSV", type=["csv"])
    if uploaded_file is not None:
        upload_path = RAW_DATA_DIR / uploaded_file.name
        upload_path.parent.mkdir(parents=True, exist_ok=True)
        with open(upload_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        input_file_path = upload_path
        st.sidebar.success(f"Saved uploaded file to {upload_path.name}")

# Target Variable
target_input = st.sidebar.text_input("Target Column", value="power_kw")

# Split Mode & Scaler
split_mode = st.sidebar.selectbox("Split Mode", VALID_SPLIT_MODES, index=0)
scaler_type = st.sidebar.selectbox("Feature Scaler", VALID_SCALERS, index=0)

# Cleaning & Feature Selection Sliders
with st.sidebar.expander("⚙️ Advanced Thresholds", expanded=False):
    missing_thresh = st.slider("Max Missingness Threshold", 0.10, 0.90, float(MISSING_COL_THRESHOLD), 0.05)
    corr_thresh = st.slider("Collinearity Threshold (|r|)", 0.70, 0.99, float(CORR_THRESHOLD), 0.02)
    iqr_mult = st.slider("IQR Multiplier for Outliers", 1.0, 3.0, float(IQR_MULTIPLIER), 0.1)
    outlier_strategy = st.selectbox("Outlier Strategy", ["cap", "remove"], index=0)
    drop_unconf = st.checkbox("Drop '*_unconfirmed' sensors", value=False)
    k_best = st.number_input("Select top K features (0 = all)", min_value=0, max_value=100, value=0)

run_button = st.sidebar.button("🚀 Run Preprocessing Pipeline", type="primary", use_container_width=True)

# Main Dashboard Layout
tab_run, tab_eda, tab_splits, tab_report = st.tabs(["🚀 Pipeline Execution", "📊 EDA Visualizations", "📁 Processed Splits", "📋 Audit Report"])

if run_button:
    if input_file_path is None or not Path(input_file_path).exists():
        st.error("Please provide a valid input CSV.")
    else:
        with st.spinner("Executing end-to-end VRP data pipeline..."):
            config = PipelineConfig(
                input_path=str(input_file_path),
                target_col=target_input,
                split_mode=split_mode,
                scaler_type=scaler_type,
                drop_unconfirmed=drop_unconf,
                outlier_strategy=outlier_strategy,
                missing_col_threshold=missing_thresh,
                corr_threshold=corr_thresh,
                iqr_multiplier=iqr_mult,
                k_best_features=k_best,
                output_dir=str(PROCESSED_DATA_DIR),
                plots_dir=str(PLOTS_DIR),
                report_path=str(OUTPUTS_DIR / "cleaning_report.json")
            )
            try:
                report = run_pipeline(config)
                st.session_state["pipeline_report"] = report
                st.success("Pipeline executed successfully!")
            except Exception as e:
                st.error(f"Pipeline error: {e}")
                st.exception(e)

# Display results if available in session_state or on disk
report_file = OUTPUTS_DIR / "cleaning_report.json"
if "pipeline_report" not in st.session_state and report_file.exists():
    try:
        with open(report_file, "r", encoding="utf-8") as f:
            st.session_state["pipeline_report"] = json.load(f)
    except Exception:
        pass

if "pipeline_report" in st.session_state:
    rep = st.session_state["pipeline_report"]

    with tab_run:
        st.subheader("📌 Preprocessing Overview & Audit Metrics")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Initial Shape", f"{rep['cleaning']['initial_rows']} x {rep['cleaning']['initial_cols']}")
        c2.metric("Cleaned Shape", f"{rep['cleaning']['final_rows']} x {rep['cleaning']['final_cols']}")
        c3.metric("Final Features", f"{rep['preprocessing']['final_feature_count']}")
        c4.metric("Elapsed Time", f"{rep['elapsed_seconds']} s")

        st.markdown("---")
        m1, m2, m3 = st.columns(3)
        with m1:
            st.info(f"**Missing-Target Dropped:** {rep['cleaning'].get('dropped_missing_target_rows', 0)}")
            st.info(f"**Duplicates Dropped:** {rep['cleaning']['duplicates_removed']}")
            st.info(f"**High-Missing Dropped:** {len(rep['cleaning']['dropped_high_missing_cols'])}")
        with m2:
            outlier_cnt = sum(v["outlier_count"] for v in rep['cleaning']['outlier_treatment'].values())
            st.info(f"**Outliers Treated:** {outlier_cnt} ({rep['configuration']['outlier_strategy']})")
            st.info(f"**Domain Guard Violations:** {sum(rep['cleaning']['domain_violations_corrected'].values())}")
        with m3:
            st.info(f"**Collinear Dropped (|r| > {rep['configuration']['corr_threshold']}):** {len(rep['feature_selection']['collinear_dropped'])}")
            st.info(f"**Split Counts:** Train {rep['splitting']['train_samples']} | Val {rep['splitting']['val_samples']} | Test {rep['splitting']['test_samples']}")

        st.subheader("🎯 Top 10 Features Correlated with Target")
        top_corr = rep["exploratory_analysis"].get("top_features_correlated_with_target", [])
        if top_corr:
            st.dataframe(pd.DataFrame(top_corr), use_container_width=True)

    with tab_eda:
        st.subheader("📊 High-Resolution Exploratory Data Analysis Charts")
        col_a, col_b = st.columns(2)
        
        missing_plot = PLOTS_DIR / "missingness_bar_chart.png"
        corr_plot = PLOTS_DIR / "correlation_heatmap.png"
        feat_target_plot = PLOTS_DIR / "feature_target_bar_plot.png"
        outlier_plot = PLOTS_DIR / "outlier_boxplots.png"

        with col_a:
            if missing_plot.exists():
                st.image(str(missing_plot), caption="Missingness Percentage per Column", use_column_width=True)
            if feat_target_plot.exists():
                st.image(str(feat_target_plot), caption="Feature Correlation with Target", use_column_width=True)

        with col_b:
            if corr_plot.exists():
                st.image(str(corr_plot), caption="Feature Pairwise Correlation Heatmap", use_column_width=True)
            if outlier_plot.exists():
                st.image(str(outlier_plot), caption="Outlier Distributions (Key Features)", use_column_width=True)

    with tab_splits:
        st.subheader("📁 Processed Dataset Splits (Zero-Leakage)")
        train_p = PROCESSED_DATA_DIR / "train.csv"
        val_p = PROCESSED_DATA_DIR / "val.csv"
        test_p = PROCESSED_DATA_DIR / "test.csv"
        job_p = PROCESSED_DATA_DIR / "preprocessor.joblib"

        col_d1, col_d2, col_d3, col_d4 = st.columns(4)
        if train_p.exists():
            with open(train_p, "rb") as f:
                col_d1.download_button("📥 Download train.csv", f.read(), "train.csv", "text/csv")
        if val_p.exists():
            with open(val_p, "rb") as f:
                col_d2.download_button("📥 Download val.csv", f.read(), "val.csv", "text/csv")
        if test_p.exists():
            with open(test_p, "rb") as f:
                col_d3.download_button("📥 Download test.csv", f.read(), "test.csv", "text/csv")
        if job_p.exists():
            with open(job_p, "rb") as f:
                col_d4.download_button("📦 Download preprocessor.joblib", f.read(), "preprocessor.joblib", "application/octet-stream")

        st.markdown("#### Sample Preview: `train.csv` (First 10 Rows)")
        if train_p.exists():
            st.dataframe(pd.read_csv(train_p, nrows=10), use_container_width=True)

    with tab_report:
        st.subheader("📋 Audit Report (`cleaning_report.json`)")
        st.json(rep)
        rep_p = OUTPUTS_DIR / "cleaning_report.json"
        if rep_p.exists():
            with open(rep_p, "rb") as f:
                st.download_button("📥 Download cleaning_report.json", f.read(), "cleaning_report.json", "application/json")
else:
    with tab_run:
        st.info("👈 Select your parameters in the sidebar and click **'Run Preprocessing Pipeline'** to start.")
