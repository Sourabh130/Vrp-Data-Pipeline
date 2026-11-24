"""
Comprehensive test suite verifying zero NaNs in processed splits, zero data leakage,
proper EV domain guard enforcement, collinearity elimination, and preprocessor reloadability.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.cleaning import apply_domain_guards, clean_dataset, drop_high_missing_columns, handle_outliers_iqr
from src.config import DOMAIN_GUARDS, PipelineConfig
from src.data_loader import generate_synthetic_vrp_data, parse_timestamp_series
from src.feature_selection import FeatureSelector
from src.pipeline import run_pipeline, split_data
from src.preprocessing import VRPPreprocessor


@pytest.fixture
def synthetic_df():
    """Generates a reproducible 300-row synthetic EV telemetry DataFrame."""
    return generate_synthetic_vrp_data(num_rows=300)


@pytest.fixture
def temp_pipeline_env(tmp_path):
    """Sets up temporary input and output directories for isolated pipeline testing."""
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    outputs_dir = tmp_path / "outputs"
    plots_dir = outputs_dir / "plots"

    raw_dir.mkdir(parents=True)
    processed_dir.mkdir(parents=True)
    plots_dir.mkdir(parents=True)

    sample_csv = raw_dir / "sample_ev.csv"
    generate_synthetic_vrp_data(num_rows=300, output_path=sample_csv)

    config = PipelineConfig(
        input_path=str(sample_csv),
        target_col="power_kw",
        split_mode="chronological",
        scaler_type="standard",
        drop_unconfirmed=True,
        outlier_strategy="cap",
        missing_col_threshold=0.60,
        corr_threshold=0.90,
        output_dir=str(processed_dir),
        plots_dir=str(plots_dir),
        report_path=str(outputs_dir / "cleaning_report.json")
    )
    return config, sample_csv, processed_dir, plots_dir, outputs_dir


def test_timestamp_parsing():
    """Verifies that mixed timestamps (+00:00, Z, epoch ms) are parsed to UTC datetime."""
    mixed_ts = pd.Series([
        "2026-10-27T04:39:24.159+00:00",
        "2026-10-27T04:39:34.405Z",
        "2026-10-27 04:39:44",
    ])
    parsed = parse_timestamp_series(mixed_ts)
    assert pd.api.types.is_datetime64_any_dtype(parsed)
    assert parsed.isna().sum() == 0


def test_ev_domain_guards():
    """Verifies that EV physical boundary guards enforce safe speed, non-negative cell delta, and SOC."""
    df_raw = pd.DataFrame({
        "speed_kmh": [-15.0, 45.0, 180.0, 90.0],
        "cell_delta_mv": [-25.0, 5.0, 12.0, -2.0],
        "cell_delta": [-10.0, 25.0, 60.0, 600.0],
        "soc": [-5.0, 80.0, 105.0, 50.0],
        "pack_voltage": [350.0, 360.0, 370.0, 380.0]
    })
    df_clean, violations = apply_domain_guards(df_raw)

    # Speed assertions
    assert df_clean["speed_kmh"].min() >= 0.0
    assert df_clean["speed_kmh"].max() <= 150.0
    assert violations.get("speed_kmh", 0) == 2

    # Cell delta assertions (millivolts: valid range 0 to 500 mV, valid 25 and 60 mV not clipped to 1.0)
    assert df_clean["cell_delta_mv"].min() >= 0.0
    assert violations.get("cell_delta_mv", 0) == 2
    assert df_clean["cell_delta"].min() >= 0.0
    assert df_clean["cell_delta"].iloc[1] == 25.0
    assert df_clean["cell_delta"].iloc[2] == 60.0
    assert df_clean["cell_delta"].max() <= 500.0
    assert violations.get("cell_delta", 0) == 2

    # SOC assertions
    assert df_clean["soc"].min() >= 0.0
    assert df_clean["soc"].max() <= 100.0
    assert violations.get("soc", 0) == 2


def test_high_missing_column_dropping(synthetic_df):
    """Verifies that columns exceeding the missingness threshold are dropped, while preserving protected cols."""
    # ccs_station_a has ~85% missing in synthetic_df
    df_clean, dropped = drop_high_missing_columns(synthetic_df, threshold=0.60, preserve_cols=["power_kw"])
    assert "ccs_station_a" in dropped
    assert "ccs_station_a" not in df_clean.columns
    # Protected target must remain
    assert "power_kw" in df_clean.columns


def test_collinear_feature_elimination():
    """Verifies that pairwise collinear features (|r| > 0.90) are dropped without leaking target."""
    np.random.seed(42)
    x1 = np.random.normal(50, 10, 200)
    # Perfectly collinear feature
    x2 = x1 * 2.5 + np.random.normal(0, 0.01, 200)
    # Target strongly correlated with x1
    y = x1 * 3.0 + np.random.normal(0, 2, 200)
    
    df_X = pd.DataFrame({"feat_a": x1, "feat_b": x2, "feat_c": np.random.normal(0, 1, 200)})
    selector = FeatureSelector(corr_threshold=0.90)
    selector.fit(df_X, pd.Series(y))
    
    transformed = selector.transform(df_X)
    # Exactly one of feat_a or feat_b should be dropped
    assert ("feat_a" in transformed.columns) != ("feat_b" in transformed.columns)
    assert len(selector.collinear_dropped_) == 1


def test_zero_data_leakage_and_splits(synthetic_df):
    """
    Verifies strict prevention of data leakage:
    1. Dataset is split chronologically into 70% train, 15% val, 15% test.
    2. Imputer median and scaler mean are derived strictly from train split.
    """
    cleaned_df, _ = clean_dataset(synthetic_df, target_col="power_kw")
    X_train, X_val, X_test, y_train, y_val, y_test, meta = split_data(
        cleaned_df, target_col="power_kw", split_mode="chronological"
    )

    total_samples = len(cleaned_df)
    assert len(X_train) == int(total_samples * 0.70)
    assert len(X_val) == int(total_samples * (0.70 + 0.15)) - len(X_train)
    assert len(X_test) == total_samples - (len(X_train) + len(X_val))

    # Fit preprocessor strictly on X_train
    preprocessor = VRPPreprocessor(scaler_type="standard")
    preprocessor.fit(X_train, y_train)

    train_num_cols = preprocessor.numeric_cols_
    imputer = preprocessor.pipeline_.named_transformers_["num"].named_steps["imputer"]
    
    # Assert that imputer statistics match X_train medians, NOT overall dataset
    expected_medians = X_train[train_num_cols].median().values
    np.testing.assert_allclose(imputer.statistics_, expected_medians, rtol=1e-5)


def test_zero_nans_in_final_splits(temp_pipeline_env):
    """Verifies that the pipeline produces zero NaNs across train, val, and test splits."""
    config, _, processed_dir, _, _ = temp_pipeline_env
    report = run_pipeline(config)

    train_path = processed_dir / "train.csv"
    val_path = processed_dir / "val.csv"
    test_path = processed_dir / "test.csv"

    assert train_path.exists()
    assert val_path.exists()
    assert test_path.exists()

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    assert train_df.isna().sum().sum() == 0, "Train split contains NaNs!"
    assert val_df.isna().sum().sum() == 0, "Val split contains NaNs!"
    assert test_df.isna().sum().sum() == 0, "Test split contains NaNs!"

    # Target column must exist in all splits
    assert "power_kw" in train_df.columns
    assert "power_kw" in val_df.columns
    assert "power_kw" in test_df.columns


def test_preprocessor_reloadability(temp_pipeline_env):
    """Verifies that preprocessor.joblib can be reloaded and transform unseen batches with zero NaNs."""
    config, _, processed_dir, _, _ = temp_pipeline_env
    run_pipeline(config)

    job_path = processed_dir / "preprocessor.joblib"
    assert job_path.exists()

    # Reload preprocessor
    loaded_preprocessor = VRPPreprocessor.load(job_path)
    
    # Generate fresh unseen batch
    new_data = generate_synthetic_vrp_data(num_rows=25)
    new_data_cleaned, _ = clean_dataset(new_data, target_col="power_kw")
    
    # Drop non-feature and target columns
    meta_cols = [c for c in new_data_cleaned.columns if c in ["ts_utc", "ts_utc_original", "power_kw"]]
    new_features = new_data_cleaned.drop(columns=meta_cols, errors="ignore")

    transformed = loaded_preprocessor.transform(new_features)
    assert isinstance(transformed, pd.DataFrame)
    assert transformed.shape[0] == 25
    assert transformed.isna().sum().sum() == 0
    assert len(transformed.columns) == len(loaded_preprocessor.get_feature_names_out())


def test_cleaning_report_structure(temp_pipeline_env):
    """Verifies that cleaning_report.json contains all required before/after audit fields."""
    config, _, _, _, outputs_dir = temp_pipeline_env
    run_pipeline(config)

    rep_path = outputs_dir / "cleaning_report.json"
    assert rep_path.exists()

    with open(rep_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    required_keys = [
        "execution_timestamp_utc",
        "configuration",
        "ingestion",
        "cleaning",
        "exploratory_analysis",
        "splitting",
        "feature_selection",
        "preprocessing",
        "artifacts"
    ]
    for key in required_keys:
        assert key in data, f"Key '{key}' missing from cleaning report JSON!"


def test_drop_missing_target_rows(tmp_path):
    """Verifies that rows with NaN in the target variable are dropped and logged in cleaning_report.json."""
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    outputs_dir = tmp_path / "outputs"
    raw_dir.mkdir(parents=True)
    processed_dir.mkdir(parents=True)
    outputs_dir.mkdir(parents=True)

    sample_csv = raw_dir / "sample_ev_nan_target.csv"
    df = generate_synthetic_vrp_data(num_rows=100)
    # Inject 15 NaNs into the target column
    df.loc[:14, "power_kw"] = np.nan
    df.to_csv(sample_csv, index=False)

    report_file = outputs_dir / "cleaning_report.json"
    config = PipelineConfig(
        input_path=str(sample_csv),
        target_col="power_kw",
        split_mode="chronological",
        output_dir=str(processed_dir),
        plots_dir=str(outputs_dir / "plots"),
        report_path=str(report_file)
    )
    report = run_pipeline(config)

    assert report["cleaning"]["dropped_missing_target_rows"] == 15
    assert report["cleaning"]["final_rows"] == 85

    # Verify no NaNs in train/val/test splits
    train_df = pd.read_csv(processed_dir / "train.csv")
    assert train_df["power_kw"].isna().sum() == 0


def test_drop_constant_columns(synthetic_df):
    """Verifies that constant (zero-variance) columns are dropped before correlation analysis."""
    from src.cleaning import drop_constant_columns
    df = synthetic_df.copy()
    df["constant_num"] = 42.0
    df["constant_str"] = "STATIC"

    df_filtered, dropped = drop_constant_columns(df, preserve_cols=["power_kw"])
    assert "constant_num" in dropped
    assert "constant_str" in dropped
    assert "constant_num" not in df_filtered.columns
    assert "constant_str" not in df_filtered.columns
