"""
features.py -- Feature selection, infinities/null handling, and normalisation.

Selects the top informative features from CIC-IDS datasets,
handles edge cases (inf, NaN), and applies RobustScaler normalisation.
"""

import os
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from typing import Optional, Tuple


# Import FEATURE_COLS from config
try:
    from src.config import FEATURE_COLS, MODEL_DIR, PROCESSED_DIR
except ImportError:
    FEATURE_COLS = []
    MODEL_DIR = "models"
    PROCESSED_DIR = "data/processed"


def select_features(df: pd.DataFrame,
                      feature_cols: Optional[list] = None) -> pd.DataFrame:
    """
    Select relevant feature columns plus metadata from a DataFrame.

    Args:
        df: Input DataFrame with CIC-IDS features.
        feature_cols: List of column names to keep as features.
                      Defaults to FEATURE_COLS from config.

    Returns:
        DataFrame with only the selected feature and metadata columns.

    Notes:
        Keeps all columns not in feature_cols that are metadata
        (timestamp, source/dest IPs, ports, protocol, stage).
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLS

    # Identify metadata columns to keep alongside features
    metadata_cols = ["Timestamp", "src_ip", "dst_ip", "src_port",
                     "dst_port", "protocol", "stage", "Label",
                     "Destination Port", "Source Port", "Protocol"]
    keep_cols = []
    for col in feature_cols:
        if col in df.columns:
            keep_cols.append(col)
    for col in metadata_cols:
        if col in df.columns and col not in keep_cols:
            keep_cols.append(col)

    available_cols = [c for c in keep_cols if c in df.columns]
    result = df[available_cols].copy()

    print(f"[features] Selected {len(available_cols)} columns "
          f"from {len(df.columns)} total")
    print(f"[features] Feature columns: {keep_cols[:5]}... ({len(keep_cols)} total)")

    return result


def handle_infinities(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace inf and -inf values with column max or 0.

    Args:
        df: DataFrame potentially containing inf/-inf values.

    Returns:
        DataFrame with inf values replaced.

    Side effects:
        Prints count of inf values replaced per column.
    """
    df = df.copy()
    for col in df.select_dtypes(include=[np.float64, np.float32]).columns:
        n_inf = np.isinf(df[col]).sum()
        if n_inf > 0:
            col_max = df.loc[~np.isinf(df[col]), col].max()
            if np.isnan(col_max) or col_max == 0:
                col_max = 0
            df.loc[np.isinf(df[col]), col] = col_max
            print(f"[features] Replaced {n_inf} inf/-inf values in '{col}' "
                  f"with max={col_max:.2f}")

    return df


def handle_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill NaN values with column medians.

    Args:
        df: DataFrame with potential NaN values.

    Returns:
        DataFrame with NaN values filled by column median.

    Side effects:
        Prints count of NaN values filled per column.
    """
    df = df.copy()
    for col in df.select_dtypes(include=[np.float64, np.float32]).columns:
        n_nan = df[col].isna().sum()
        if n_nan > 0:
            col_median = df[col].median()
            if np.isnan(col_median):
                col_median = 0
            df[col] = df[col].fillna(col_median)
            print(f"[features] Filled {n_nan} NaN values in '{col}' "
                  f"with median={col_median:.2f}")

    return df


def normalise(df: pd.DataFrame,
               scaler: Optional[RobustScaler] = None) -> Tuple[pd.DataFrame, RobustScaler]:
    """
    Normalise feature columns using RobustScaler.

    RobustScaler is preferred for network traffic data because
    it uses median and IQR rather than mean and standard deviation,
    making it robust to outliers.

    Args:
        df: DataFrame with feature columns to normalise.
        scaler: Optional pre-fitted RobustScaler. If None, fits a new one.

    Returns:
        Tuple of (normalised DataFrame, fitted RobustScaler).

    Side effects:
        Saves the fitted scaler to data/processed/scaler.pkl.
    """
    if scaler is None:
        scaler = RobustScaler()

    feature_cols = [c for c in FEATURE_COLS if c in df.columns]
    if not feature_cols:
        print("[features] WARNING: No feature columns found for normalisation")
        return df, scaler

    # Fit or transform
    df_normalised = df.copy()
    df_normalised[feature_cols] = scaler.fit_transform(df[feature_cols])

    # Save scaler
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    scaler_path = os.path.join(PROCESSED_DIR, "scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"[features] Scaler saved to {scaler_path}")

    return df_normalised, scaler


if __name__ == "__main__":
    print("=== features.py self-test ===")
    print("Imported successfully -- waiting for pipeline integration")
