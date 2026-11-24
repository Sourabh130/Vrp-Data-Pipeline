"""
Master pipeline runner executing end-to-end VRP data cleaning, exploratory analysis,
leak-free feature selection, scaling, and splitting for JEV model consumption.
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Ensure project root is in sys.path for CLI execution
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
import pandas as pd
from sklearn.model_selection._split import train_test_split

from src.analysis import run_full_analysis
from src.cleaning import clean_dataset, drop_constant_columns, to_snake_case
from src.config import (
    CLEANING_REPORT_PATH,
    CORR_THRESHOLD,
    DEFAULT_RAW_FILE,
    IQR_MULTIPLIER,
    METADATA_COLUMNS,
    MISSING_COL_THRESHOLD,
    PLOTS_DIR,
    PREPROCESSOR_JOB_PATH,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
    SPLIT_RATIOS,
    VALID_SCALERS,
    VALID_SPLIT_MODES,
    PipelineConfig,
)
from src.data_loader import load_dataset
from src.feature_selection import FeatureSelector
from src.preprocessing import VRPPreprocessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("vrp_pipeline")


def split_data(
    df: pd.DataFrame,
    target_col: str,
    split_mode: str = "chronological",
    ratios: Tuple[float, float, float] = SPLIT_RATIOS,
    random_state: int = RANDOM_STATE
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, Dict[str, Any]]:
    """
    Splits the dataset into train (70%), val (15%), test (15%)
    strictly BEFORE feature selection, imputation, encoding, and scaling.
    """
    total_len = len(df)
    train_pct, val_pct, test_pct = ratios
    
    # Identify non-feature columns
    metadata_cols = [c for c in df.columns if c in METADATA_COLUMNS or c.startswith("__")]
    feature_cols = [c for c in df.columns if c != target_col and c not in metadata_cols]

    X = df[feature_cols].copy()
    y = df[target_col].copy()

    split_info = {
        "split_mode": split_mode,
        "train_ratio": train_pct,
        "val_ratio": val_pct,
        "test_ratio": test_pct,
        "total_samples": total_len,
        "metadata_excluded_from_features": metadata_cols
    }

    if split_mode == "chronological":
        # Preserve temporal ordering
        train_idx_end = int(total_len * train_pct)
        val_idx_end = int(total_len * (train_pct + val_pct))

        X_train, y_train = X.iloc[:train_idx_end], y.iloc[:train_idx_end]
        X_val, y_val = X.iloc[train_idx_end:val_idx_end], y.iloc[train_idx_end:val_idx_end]
        X_test, y_test = X.iloc[val_idx_end:], y.iloc[val_idx_end:]

        split_info["train_range"] = (0, train_idx_end)
        split_info["val_range"] = (train_idx_end, val_idx_end)
        split_info["test_range"] = (val_idx_end, total_len)
    else:
        # Random split
        val_test_ratio = val_pct + test_pct
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=val_test_ratio, random_state=random_state, shuffle=True
        )
        # Split temp into 50% val and 50% test (15% each of original)
        val_relative_ratio = val_pct / val_test_ratio
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, train_size=val_relative_ratio, random_state=random_state, shuffle=True
        )

    split_info["train_samples"] = len(X_train)
    split_info["val_samples"] = len(X_val)
    split_info["test_samples"] = len(X_test)

    logger.info(
        f"Data split ({split_mode}): Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)} samples."
    )

    return X_train, X_val, X_test, y_train, y_val, y_test, split_info


def run_pipeline(config: PipelineConfig) -> Dict[str, Any]:
    """
    Executes the complete VRP ML data cleaning and preprocessing pipeline end-to-end.
    """
    start_time = datetime.utcnow()
    logger.info(f"Starting VRP pipeline with input: {config.input_path}")
    
    # 1. Ingestion
    raw_df, loader_meta = load_dataset(config.input_path, auto_generate_if_missing=True)
    
    # If split mode is chronological but no timestamp is present, log note
    split_mode = config.split_mode
    if split_mode == "chronological" and not loader_meta["has_timestamp"]:
        logger.info("No explicit timestamp column detected; applying index-based chronological split.")

    # 2. Cleaning & Domain Guards & Outlier Handling
    cleaned_df, cleaning_audit = clean_dataset(
        df=raw_df,
        target_col=config.target_col,
        missing_col_threshold=config.missing_col_threshold,
        drop_unconfirmed=config.drop_unconfirmed,
        iqr_multiplier=config.iqr_multiplier,
        outlier_strategy=config.outlier_strategy,
        extra_metadata_cols=config.extra_metadata_cols
    )

    target_clean = cleaning_audit["target_column"]
    if target_clean not in cleaned_df.columns:
        raise ValueError(
            f"Target column '{config.target_col}' (cleaned as '{target_clean}') was not found in dataset! "
            f"Available columns: {list(cleaned_df.columns)}"
        )

    # 3. Handle missing target rows (drop rows where target is NaN instead of imputing)
    missing_target_count = int(cleaned_df[target_clean].isna().sum())
    if missing_target_count > 0:
        logger.info(f"Dropping {missing_target_count} rows with NaN in target '{target_clean}'.")
        cleaned_df = cleaned_df.dropna(subset=[target_clean]).reset_index(drop=True)
    cleaning_audit["dropped_missing_target_rows"] = missing_target_count

    # 4. Drop constant (zero-variance) columns before running correlation analysis
    protected = set([target_clean, "ts_utc", "ts_utc_original", "ts", "timestamp"] + (config.extra_metadata_cols or []))
    cleaned_df, constant_cols = drop_constant_columns(cleaned_df, preserve_cols=list(protected))
    if constant_cols:
        existing_const = set(cleaning_audit.get("dropped_constant_cols", []))
        cleaning_audit["dropped_constant_cols"] = sorted(list(existing_const.union(constant_cols)))

    cleaning_audit["final_rows"] = int(cleaned_df.shape[0])
    cleaning_audit["final_cols"] = int(cleaned_df.shape[1])

    # 5. Exploratory Analysis & High-Res PNG Visualizations
    analysis_meta = run_full_analysis(
        df=cleaned_df,
        target_col=target_clean,
        plots_dir=Path(config.plots_dir),
        missing_threshold=config.missing_col_threshold
    )

    # 6. Data Splitting (Before fitting selectors, imputers, scalers)
    X_train, X_val, X_test, y_train, y_val, y_test, split_meta = split_data(
        df=cleaned_df,
        target_col=target_clean,
        split_mode=split_mode,
        ratios=config.split_ratios,
        random_state=config.random_state
    )

    # 5. Feature Selection (Fitted STRICTLY on X_train to prevent leakage)
    feature_selector = FeatureSelector(
        corr_threshold=config.corr_threshold,
        k_best=config.k_best_features,
        random_state=config.random_state
    )
    feature_selector.fit(X_train, y_train)
    
    X_train_sel = feature_selector.transform(X_train)
    X_val_sel = feature_selector.transform(X_val)
    X_test_sel = feature_selector.transform(X_test)

    feature_sel_summary = feature_selector.get_summary()

    # 6. Preprocessing: Imputation + Scaling + One-Hot Encoding (Fitted STRICTLY on X_train_sel)
    preprocessor = VRPPreprocessor(scaler_type=config.scaler_type)
    preprocessor.fit(X_train_sel, y_train)

    X_train_proc = preprocessor.transform(X_train_sel)
    X_val_proc = preprocessor.transform(X_val_sel)
    X_test_proc = preprocessor.transform(X_test_sel)

    # Verify zero NaNs in processed feature sets
    train_nans = int(X_train_proc.isna().sum().sum())
    val_nans = int(X_val_proc.isna().sum().sum())
    test_nans = int(X_test_proc.isna().sum().sum())
    if train_nans + val_nans + test_nans > 0:
        logger.warning(f"Unexpected NaNs after preprocessing: Train={train_nans}, Val={val_nans}, Test={test_nans}")

    # 7. Construct Final Processed Splits (Features + Target)
    train_df = X_train_proc.copy()
    train_df[target_clean] = y_train.values

    val_df = X_val_proc.copy()
    val_df[target_clean] = y_val.values

    test_df = X_test_proc.copy()
    test_df[target_clean] = y_test.values

    # 8. Save Processed Splits and Preprocessor Artifact
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = out_dir / "train.csv"
    val_path = out_dir / "val.csv"
    test_path = out_dir / "test.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    preprocessor_path = Path(config.output_dir) / "preprocessor.joblib"
    preprocessor.save(preprocessor_path)

    # 9. Build Structured Audit Report
    elapsed_seconds = (datetime.utcnow() - start_time).total_seconds()
    
    # Calculate top correlated features with target from cleaned dataset
    top_target_corr = analysis_meta.get("top_features_correlated_with_target", [])

    report = {
        "execution_timestamp_utc": start_time.isoformat() + "Z",
        "elapsed_seconds": round(elapsed_seconds, 2),
        "configuration": {
            "input_path": str(config.input_path),
            "target_column": target_clean,
            "split_mode": split_mode,
            "scaler_type": config.scaler_type,
            "drop_unconfirmed": config.drop_unconfirmed,
            "missing_col_threshold": config.missing_col_threshold,
            "corr_threshold": config.corr_threshold,
            "iqr_multiplier": config.iqr_multiplier,
            "outlier_strategy": config.outlier_strategy,
            "k_best_features": config.k_best_features,
        },
        "ingestion": loader_meta,
        "cleaning": cleaning_audit,
        "exploratory_analysis": {
            "missingness_summary": analysis_meta.get("missingness_summary", {}),
            "top_features_correlated_with_target": top_target_corr,
            "plots_generated": analysis_meta.get("plots_saved", []),
        },
        "splitting": split_meta,
        "feature_selection": feature_sel_summary,
        "preprocessing": {
            "scaler_type": config.scaler_type,
            "numeric_columns_count": len(preprocessor.numeric_cols_),
            "categorical_columns_count": len(preprocessor.categorical_cols_),
            "final_feature_count": len(preprocessor.feature_names_out_),
            "final_feature_names": preprocessor.feature_names_out_,
            "post_processing_nans": {
                "train": train_nans,
                "val": val_nans,
                "test": test_nans,
            }
        },
        "artifacts": {
            "train_csv": str(train_path),
            "val_csv": str(val_path),
            "test_csv": str(test_path),
            "preprocessor_joblib": str(preprocessor_path),
            "plots_dir": str(config.plots_dir),
        }
    }

    # Save cleaning report JSON
    rep_path = Path(config.report_path)
    rep_path.parent.mkdir(parents=True, exist_ok=True)
    with open(rep_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Saved structured cleaning report to {rep_path}")

    # Print summary to console
    print("\n" + "=" * 70)
    print("           VRP DATA PIPELINE EXECUTION SUMMARY")
    print("=" * 70)
    print(f"  Input Dataset        : {config.input_path}")
    print(f"  Target Variable      : {target_clean}")
    print(f"  Initial Shape        : {cleaning_audit['initial_rows']} rows x {cleaning_audit['initial_cols']} cols")
    print(f"  Cleaned Shape        : {cleaning_audit['final_rows']} rows x {cleaning_audit['final_cols']} cols")
    print(f"  Dropped Missing-Target: {cleaning_audit.get('dropped_missing_target_rows', 0)} rows")
    print(f"  Dropped High-Missing : {list(cleaning_audit['dropped_high_missing_cols'].keys())}")
    print(f"  Dropped Constant     : {cleaning_audit.get('dropped_constant_cols', [])}")
    print(f"  Dropped Collinear    : {feature_sel_summary['collinear_dropped']}")
    print(f"  Domain Violations    : {cleaning_audit['domain_violations_corrected']}")
    total_outliers = sum(v["outlier_count"] for v in cleaning_audit["outlier_treatment"].values())
    print(f"  Outliers Treated     : {total_outliers} (strategy: {config.outlier_strategy})")
    print(f"  Split Mode & Counts  : {split_mode} -> Train: {split_meta['train_samples']}, Val: {split_meta['val_samples']}, Test: {split_meta['test_samples']}")
    print(f"  Final Feature Count  : {len(preprocessor.feature_names_out_)}")
    print(f"  Zero NaNs Confirmed  : Train={train_nans}, Val={val_nans}, Test={test_nans}")
    print("\n  Top 10 Correlated Features with Target:")
    for item in top_target_corr[:10]:
        print(f"    - {item['feature']:<25}: r = {item['pearson_r']:.4f}")
    print("=" * 70)
    print(f"  Artifacts saved:")
    print(f"    - {train_path}")
    print(f"    - {val_path}")
    print(f"    - {test_path}")
    print(f"    - {preprocessor_path}")
    print(f"    - {rep_path}")
    print("=" * 70 + "\n")

    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Production-Grade VRP / EV Telemetry Data Cleaning & Preprocessing Pipeline"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(DEFAULT_RAW_FILE),
        help="Path to single CSV, directory, or glob pattern of raw VRP CSVs."
    )
    parser.add_argument(
        "--target",
        type=str,
        default="power_kw",
        help="Target column name (e.g. power_kw)."
    )
    parser.add_argument(
        "--split-mode",
        type=str,
        choices=VALID_SPLIT_MODES,
        default="chronological",
        help="Data split strategy: 'chronological' (time-series safe) or 'random'."
    )
    parser.add_argument(
        "--scaler",
        type=str,
        choices=VALID_SCALERS,
        default="standard",
        help="Scaler type for numerical features: 'standard', 'minmax', or 'robust'."
    )
    parser.add_argument(
        "--drop-unconfirmed",
        action="store_true",
        help="Flag to drop '*_unconfirmed' sensor columns (useful for voltlens-*.csv files)."
    )
    parser.add_argument(
        "--outlier-strategy",
        type=str,
        choices=["cap", "remove"],
        default="cap",
        help="Strategy for handling IQR outliers: 'cap' (winsorize) or 'remove' (drop rows)."
    )
    parser.add_argument(
        "--missing-threshold",
        type=float,
        default=MISSING_COL_THRESHOLD,
        help="Missing value ratio threshold to drop columns (default: 0.60)."
    )
    parser.add_argument(
        "--corr-threshold",
        type=float,
        default=CORR_THRESHOLD,
        help="Pairwise correlation threshold to drop collinear features (default: 0.90)."
    )
    parser.add_argument(
        "--iqr-multiplier",
        type=float,
        default=IQR_MULTIPLIER,
        help="IQR multiplier for outlier detection (default: 1.5)."
    )
    parser.add_argument(
        "--k-best",
        type=int,
        default=0,
        help="Select top K features using SelectKBest (default: 0 = retain all non-collinear features)."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROCESSED_DATA_DIR),
        help="Output directory for processed CSV splits and preprocessor.joblib."
    )
    parser.add_argument(
        "--plots-dir",
        type=str,
        default=str(PLOTS_DIR),
        help="Output directory for generated PNG charts."
    )
    parser.add_argument(
        "--report-path",
        type=str,
        default=str(CLEANING_REPORT_PATH),
        help="Path to save the structured cleaning report JSON."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = PipelineConfig(
        input_path=args.input,
        target_col=args.target,
        split_mode=args.split_mode,
        scaler_type=args.scaler,
        drop_unconfirmed=args.drop_unconfirmed,
        outlier_strategy=args.outlier_strategy,
        missing_col_threshold=args.missing_threshold,
        corr_threshold=args.corr_threshold,
        iqr_multiplier=args.iqr_multiplier,
        k_best_features=args.k_best,
        output_dir=args.output_dir,
        plots_dir=args.plots_dir,
        report_path=args.report_path
    )
    try:
        run_pipeline(config)
    except Exception as e:
        logger.exception(f"Pipeline execution failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
