"""
Feature selection module to drop zero-variance features, eliminate multicollinearity,
and select top predictive features strictly fitted on the training split.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection._univariate_selection import SelectKBest, f_regression

from src.config import CORR_THRESHOLD

logger = logging.getLogger(__name__)


class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    Production-grade scikit-learn compatible feature selector.
    Fitted strictly on X_train to prevent data leakage.
    
    Stages:
    1. Removes constant / zero-variance features.
    2. Identifies and removes highly collinear features (|r| > corr_threshold).
    3. (Optional) Selects top K features using SelectKBest.
    """

    def __init__(
        self,
        corr_threshold: float = CORR_THRESHOLD,
        k_best: int = 0,
        score_func: str = "f_regression",
        random_state: int = 42
    ):
        self.corr_threshold = corr_threshold
        self.k_best = k_best
        self.score_func = score_func
        self.random_state = random_state
        
        # Fitted attributes
        self.zero_variance_cols_: List[str] = []
        self.collinear_pairs_: List[Dict[str, Any]] = []
        self.collinear_dropped_: List[str] = []
        self.k_best_dropped_: List[str] = []
        self.selected_features_: List[str] = []
        self.feature_scores_: Dict[str, float] = {}

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Fits the feature selector on X_train and optional y_train.
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        current_cols = list(X.columns)

        # 1. Identify zero-variance features
        self.zero_variance_cols_ = [
            c for c in current_cols
            if X[c].nunique(dropna=True) <= 1
            or (pd.api.types.is_numeric_dtype(X[c]) and len(X[c].dropna()) > 0 and X[c].dropna().std() <= 1e-12)
        ]
        if self.zero_variance_cols_:
            logger.info(f"Identified {len(self.zero_variance_cols_)} zero-variance columns: {self.zero_variance_cols_}")
        
        remaining_cols = [c for c in current_cols if c not in self.zero_variance_cols_]

        # 2. Identify collinear features among numeric columns
        numeric_df = X[remaining_cols].select_dtypes(include=[np.number])
        valid_num_cols = [
            c for c in numeric_df.columns
            if numeric_df[c].nunique(dropna=True) > 1 and numeric_df[c].std() > 1e-12
        ]
        numeric_df = numeric_df[valid_num_cols]
        if len(numeric_df.columns) > 1:
            corr_matrix = numeric_df.corr().abs()
            
            # If target y is provided, compute feature-target correlations to break ties intelligently
            target_corr = {}
            if y is not None:
                y_clean = pd.to_numeric(y, errors="coerce")
                for col in numeric_df.columns:
                    valid_idx = numeric_df[col].notna() & y_clean.notna()
                    if valid_idx.sum() > 5:
                        r = np.corrcoef(numeric_df.loc[valid_idx, col], y_clean.loc[valid_idx])[0, 1]
                        target_corr[col] = abs(r) if not np.isnan(r) else 0.0
                    else:
                        target_corr[col] = 0.0

            dropped_collinear = set()
            col_list = list(numeric_df.columns)
            
            for i in range(len(col_list)):
                col_i = col_list[i]
                if col_i in dropped_collinear:
                    continue
                for j in range(i + 1, len(col_list)):
                    col_j = col_list[j]
                    if col_j in dropped_collinear:
                        continue
                    
                    r_val = corr_matrix.loc[col_i, col_j]
                    if r_val > self.corr_threshold:
                        # Decide which one to drop
                        # If target_corr available, keep feature with higher correlation to target
                        score_i = target_corr.get(col_i, 0.0)
                        score_j = target_corr.get(col_j, 0.0)
                        
                        to_drop = col_j if score_i >= score_j else col_i
                        to_keep = col_i if to_drop == col_j else col_j
                        
                        dropped_collinear.add(to_drop)
                        self.collinear_pairs_.append({
                            "feature_1": col_i,
                            "feature_2": col_j,
                            "correlation": round(float(r_val), 4),
                            "dropped": to_drop,
                            "retained": to_keep
                        })
                        
                        if to_drop == col_i:
                            break  # col_i is dropped, move to next i

            self.collinear_dropped_ = sorted(list(dropped_collinear))
            if self.collinear_dropped_:
                logger.info(
                    f"Dropped {len(self.collinear_dropped_)} collinear columns (|r| > {self.corr_threshold}): "
                    f"{self.collinear_dropped_}"
                )

        # Update remaining columns
        post_collinear_cols = [
            c for c in remaining_cols
            if c not in self.collinear_dropped_
        ]

        # 3. Optional SelectKBest for numeric features
        if self.k_best > 0 and y is not None and len(post_collinear_cols) > self.k_best:
            numeric_post = [c for c in post_collinear_cols if c in numeric_df.columns]
            non_numeric = [c for c in post_collinear_cols if c not in numeric_df.columns]
            
            if len(numeric_post) > self.k_best:
                # Impute temporarily for scoring
                X_num_imp = X[numeric_post].fillna(X[numeric_post].median())
                y_imp = pd.to_numeric(y, errors="coerce").fillna(y.median())
                
                func = f_regression
                scores, _ = func(X_num_imp, y_imp)
                
                score_dict = dict(zip(numeric_post, scores))
                self.feature_scores_ = {k: round(float(v), 4) for k, v in score_dict.items() if not np.isnan(v)}
                
                # Sort and select top K
                sorted_feats = sorted(self.feature_scores_.keys(), key=lambda k: self.feature_scores_[k], reverse=True)
                selected_num = sorted_feats[:self.k_best]
                self.k_best_dropped_ = [c for c in numeric_post if c not in selected_num]
                self.selected_features_ = selected_num + non_numeric
            else:
                self.selected_features_ = post_collinear_cols
        else:
            self.selected_features_ = post_collinear_cols

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms dataset by retaining only selected features.
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        return X[[c for c in self.selected_features_ if c in X.columns]].copy()

    def get_feature_names_out(self, input_features=None) -> List[str]:
        return list(self.selected_features_)

    def get_summary(self) -> Dict[str, Any]:
        return {
            "zero_variance_dropped": self.zero_variance_cols_,
            "collinear_dropped": self.collinear_dropped_,
            "collinear_pairs": self.collinear_pairs_,
            "k_best_dropped": self.k_best_dropped_,
            "selected_features_count": len(self.selected_features_),
            "selected_features": self.selected_features_
        }
