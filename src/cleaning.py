"""
Data cleaning module for column standardization, deduplication, missing value filtering,
EV domain guard enforcement, and IQR outlier treatment.
"""

import logging
import re
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

from src.config import (
    DOMAIN_GUARDS,
    IQR_MULTIPLIER,
    MISSING_COL_THRESHOLD,
    METADATA_COLUMNS,
)

logger = logging.getLogger(__name__)


def to_snake_case(col_name: str) -> str:
    """
    Standardizes a column name to snake_case.
    Removes extraneous punctuation and leading/trailing underscores.
    """
    s = str(col_name).strip()
    # Replace whitespace and hyphens with underscores
    s = re.sub(r"[\s\-]+", "_", s)
    # Convert camelCase to snake_case
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s)
    # Remove characters that are not alphanumeric or underscore
    s = re.sub(r"[^\w\.]", "_", s)
    # Collapse multiple underscores
    s = re.sub(r"_+", "_", s)
    # Strip leading/trailing underscores unless original was purely special
    s = s.strip("_").lower()
    return s if s else "col"


def standardize_column_names(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Renames all columns to snake_case and returns the modified DataFrame and rename mapping.
    """
    mapping = {}
    new_cols = []
    seen = {}
    
    for c in df.columns:
        clean = to_snake_case(c)
        if clean in seen:
            seen[clean] += 1
            clean = f"{clean}_{seen[clean]}"
        else:
            seen[clean] = 0
        mapping[c] = clean
        new_cols.append(clean)
        
    df_clean = df.copy()
    df_clean.columns = new_cols
    return df_clean, mapping


def drop_unconfirmed_sensor_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Drops sensor columns containing '_unconfirmed'.
    """
    unconfirmed_cols = [c for c in df.columns if "_unconfirmed" in c or c.endswith("_unconfirmed")]
    df_filtered = df.drop(columns=unconfirmed_cols, errors="ignore")
    return df_filtered, unconfirmed_cols


def drop_high_missing_columns(
    df: pd.DataFrame,
    threshold: float = MISSING_COL_THRESHOLD,
    preserve_cols: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Drops columns whose missing percentage strictly exceeds threshold.
    Preserves protected columns (e.g. target, timestamps).
    """
    preserve = set(preserve_cols or [])
    missing_pct = df.isnull().mean()
    dropped = {}
    
    for col, pct in missing_pct.items():
        if pct > threshold and col not in preserve:
            dropped[col] = round(float(pct), 4)
            
    df_filtered = df.drop(columns=list(dropped.keys()))
    return df_filtered, dropped


def drop_constant_columns(
    df: pd.DataFrame,
    preserve_cols: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Identifies and drops columns with zero variance (constant columns where nunique <= 1, ignoring NaNs).
    Preserves protected columns (e.g. target, timestamps).
    """
    preserve = set(preserve_cols or [])
    dropped = []
    
    for col in df.columns:
        if col not in preserve:
            if df[col].nunique(dropna=True) <= 1:
                dropped.append(col)
                
    df_filtered = df.drop(columns=dropped)
    if dropped:
        logger.info(f"Dropped {len(dropped)} constant/zero-variance columns: {dropped}")
    return df_filtered, dropped


def apply_domain_guards(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Enforces EV physical domain guards:
    - Speed cannot be negative or > 150 km/h
    - Cell delta voltage (cell_delta_mv, cell_delta) valid range 0 to 500 mV (cannot be negative)
    - SOC cannot be outside [0, 100]
    - Pack voltage cannot be <= 0
    """
    df_clean = df.copy()
    violations: Dict[str, int] = {}

    for col, guard in DOMAIN_GUARDS.items():
        if col in df_clean.columns and pd.api.types.is_numeric_dtype(df_clean[col]):
            s = df_clean[col]
            v_count = 0
            
            # Minimum check
            if "min" in guard:
                min_val = guard["min"]
                below_mask = (s < min_val) & s.notna()
                cnt = int(below_mask.sum())
                if cnt > 0:
                    v_count += cnt
                    df_clean.loc[below_mask, col] = min_val
                    
            # Maximum check
            if "max" in guard:
                max_val = guard["max"]
                above_mask = (s > max_val) & s.notna()
                cnt = int(above_mask.sum())
                if cnt > 0:
                    v_count += cnt
                    df_clean.loc[above_mask, col] = max_val
                    
            if v_count > 0:
                violations[col] = v_count
                logger.info(f"Domain guard enforced on '{col}': corrected {v_count} out-of-bound values.")

    return df_clean, violations


def handle_outliers_iqr(
    df: pd.DataFrame,
    multiplier: float = IQR_MULTIPLIER,
    strategy: str = "cap",
    exclude_cols: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
    """
    Detects and handles numerical outliers using the IQR rule (Q1 - 1.5*IQR, Q3 + 1.5*IQR).
    strategy='cap': Winsorizes / clips values to bounds.
    strategy='remove': Drops rows containing outliers.
    """
    df_out = df.copy()
    exclude = set(exclude_cols or [])
    outlier_stats: Dict[str, Dict[str, Any]] = {}
    rows_to_drop = set()

    # Identify continuous numeric columns
    numeric_cols = [
        c for c in df_out.select_dtypes(include=[np.number]).columns
        if c not in exclude and df_out[c].nunique() > 2
    ]

    for col in numeric_cols:
        series = df_out[col].dropna()
        if len(series) < 10:
            continue
            
        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        iqr = q3 - q1
        
        if iqr <= 1e-8:
            continue  # Low variability column
            
        lower_bound = q1 - multiplier * iqr
        upper_bound = q3 + multiplier * iqr
        
        lower_mask = df_out[col] < lower_bound
        upper_mask = df_out[col] > upper_bound
        outlier_count = int((lower_mask | upper_mask).sum())
        
        if outlier_count > 0:
            outlier_stats[col] = {
                "q1": round(q1, 4),
                "q3": round(q3, 4),
                "iqr": round(iqr, 4),
                "lower_bound": round(lower_bound, 4),
                "upper_bound": round(upper_bound, 4),
                "outlier_count": outlier_count,
                "strategy_applied": strategy
            }
            
            if strategy == "cap":
                df_out[col] = df_out[col].clip(lower=lower_bound, upper=upper_bound)
            elif strategy == "remove":
                outlier_indices = df_out.index[lower_mask | upper_mask]
                rows_to_drop.update(outlier_indices)

    if strategy == "remove" and rows_to_drop:
        logger.info(f"Removing {len(rows_to_drop)} rows with IQR outliers.")
        df_out = df_out.drop(index=list(rows_to_drop)).reset_index(drop=True)

    return df_out, outlier_stats


def clean_dataset(
    df: pd.DataFrame,
    target_col: str = "power_kw",
    missing_col_threshold: float = MISSING_COL_THRESHOLD,
    drop_unconfirmed: bool = False,
    iqr_multiplier: float = IQR_MULTIPLIER,
    outlier_strategy: str = "cap",
    extra_metadata_cols: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Executes the complete cleaning stage:
    1. Standardizes column names
    2. Drops unconfirmed columns if requested
    3. Drops exact duplicates
    4. Drops columns with missingness > missing_col_threshold
    5. Applies physical domain guards
    6. Handles outliers via IQR
    """
    audit = {
        "initial_rows": int(df.shape[0]),
        "initial_cols": int(df.shape[1]),
        "duplicates_removed": 0,
        "dropped_unconfirmed_cols": [],
        "dropped_high_missing_cols": {},
        "dropped_constant_cols": [],
        "domain_violations_corrected": {},
        "outlier_treatment": {},
        "final_rows": 0,
        "final_cols": 0,
    }

    # 1. Standardize column names
    df_cleaned, col_mapping = standardize_column_names(df)
    target_clean = to_snake_case(target_col)

    # 2. Drop unconfirmed sensor columns if requested
    if drop_unconfirmed:
        df_cleaned, unconf_cols = drop_unconfirmed_sensor_columns(df_cleaned)
        audit["dropped_unconfirmed_cols"] = unconf_cols

    # 3. Drop exact duplicates
    rows_before_dedup = len(df_cleaned)
    df_cleaned = df_cleaned.drop_duplicates().reset_index(drop=True)
    dedup_count = rows_before_dedup - len(df_cleaned)
    audit["duplicates_removed"] = int(dedup_count)

    # 4. Drop columns exceeding missing threshold
    protected = set([target_clean, "ts_utc", "ts_utc_original", "ts", "timestamp"] + (extra_metadata_cols or []))
    df_cleaned, dropped_missing = drop_high_missing_columns(
        df_cleaned,
        threshold=missing_col_threshold,
        preserve_cols=list(protected)
    )
    audit["dropped_high_missing_cols"] = dropped_missing

    # 5. Apply EV domain guards
    df_cleaned, domain_violations = apply_domain_guards(df_cleaned)
    audit["domain_violations_corrected"] = domain_violations

    # 6. Outlier detection & treatment
    non_feature_exclusions = list(METADATA_COLUMNS) + [target_clean] + (extra_metadata_cols or [])
    df_cleaned, outlier_stats = handle_outliers_iqr(
        df_cleaned,
        multiplier=iqr_multiplier,
        strategy=outlier_strategy,
        exclude_cols=non_feature_exclusions
    )
    audit["outlier_treatment"] = outlier_stats

    # 7. Drop constant (zero-variance) columns
    df_cleaned, dropped_constant = drop_constant_columns(
        df_cleaned,
        preserve_cols=list(protected)
    )
    audit["dropped_constant_cols"] = dropped_constant

    audit["final_rows"] = int(df_cleaned.shape[0])
    audit["final_cols"] = int(df_cleaned.shape[1])
    audit["target_column"] = target_clean

    logger.info(
        f"Cleaning complete: {audit['initial_rows']} -> {audit['final_rows']} rows, "
        f"{audit['initial_cols']} -> {audit['final_cols']} cols."
    )

    return df_cleaned, audit
