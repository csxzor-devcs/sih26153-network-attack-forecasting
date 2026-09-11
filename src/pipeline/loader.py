"""
loader.py -- CSV loading and dataset merging for SIH26153.

Loads CIC-IDS2017 and UNSW-NB15 raw CSV files with memory-efficient
dtype specifications, cleans column names, and concatenates datasets.
"""

import os
import glob
import pandas as pd
import numpy as np
from typing import Optional


# Specify dtypes to minimize memory usage
_CIC_DTYPES = {
    "Flow Duration": "float64",
    "Total Fwd Packets": "int64",
    "Total Backward Packets": "int64",
    "Total Length of Fwd Packets": "int64",
    "Total Length of Bwd Packets": "int64",
    "Fwd Packet Length Max": "int64",
    "Fwd Packet Length Min": "int64",
    "Fwd Packet Length Mean": "float64",
    "Fwd Packet Length Std": "float64",
    "Bwd Packet Length Max": "int64",
    "Bwd Packet Length Min": "int64",
    "Bwd Packet Length Mean": "float64",
    "Bwd Packet Length Std": "float64",
    "Flow Bytes/s": "float64",
    "Flow Packets/s": "float64",
    "Flow IAT Mean": "float64",
    "Flow IAT Std": "float64",
    "Flow IAT Max": "float64",
    "Flow IAT Min": "float64",
    "Fwd IAT Total": "float64",
    "Fwd IAT Mean": "float64",
    "Fwd IAT Std": "float64",
    "Fwd IAT Max": "float64",
    "Fwd IAT Min": "float64",
    "Bwd IAT Total": "float64",
    "Bwd IAT Mean": "float64",
    "Bwd IAT Std": "float64",
    "Bwd IAT Max": "float64",
    "Bwd IAT Min": "float64",
    "Fwd Header Length": "int64",
    "Bwd Header Length": "int64",
    "Fwd Packets/s": "float64",
    "Bwd Packets/s": "float64",
    "Packet Length Mean": "float64",
    "Packet Length Std": "float64",
    "Packet Length Max": "int64",
    "Packet Length Min": "int64",
    "SYN Flag Count": "int64",
    "FIN Flag Count": "int64",
    "PSH Flag Count": "int64",
    "ACK Flag Count": "int64",
    "URG Flag Count": "int64",
    "Average Packet Size": "float64",
    "Fwd Segment Size": "int64",
    "Bwd Segment Size": "int64",
    "Init_Win_bytes_forward": "int64",
    "Init_Win_bytes_backward": "int64",
    "act_data_pkt_fwd": "int64",
    "min_seg_size_forward": "int64",
    "Active Mean": "float64",
    "Active Std": "float64",
    "Active Max": "float64",
    "Active Min": "float64",
    "Idle Mean": "float64",
    "Idle Std": "float64",
    "Idle Max": "float64",
    "Idle Min": "float64",
    "CWE Flag Count": "int64",
    "ECE Flag Count": "int64",
    "Down/Up Ratio": "float64",
    "Destination Port": "int64",
    "Source Port": "int64",
    "Protocol": "object",
    "Timestamp": "object",
    "Label": "object",
}


def load_cicids2017(data_dir: str, max_rows: Optional[int] = None) -> pd.DataFrame:
    """
    Load all CIC-IDS2017 CSV files from a directory with memory-efficient dtypes.

    Args:
        data_dir: Path to directory containing CIC-IDS2017 CSV files.
        max_rows: Maximum total rows to load across all files.
                  If None, loads all rows.

    Returns:
        Concatenated DataFrame with cleaned column names and string values.
    """
    csv_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    dfs = []
    total_loaded = 0
    for f in csv_files:
        df = pd.read_csv(f, low_memory=False, dtype=_CIC_DTYPES)
        dfs.append(df)
        total_loaded += len(df)
        print(f"[loader] Loaded {len(df)} rows from {os.path.basename(f)}")

    df = pd.concat(dfs, ignore_index=True)
    print(f"[loader] Total rows before cleanup: {len(df)}")

    # Strip whitespace from column names
    df.columns = df.columns.str.strip()

    # Strip whitespace from string columns
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()

    # Drop rows where Label is NaN
    before = len(df)
    df = df.dropna(subset=["Label"])
    after = len(df)
    print(f"[loader] Dropped {before - after} rows with NaN Label")

    # Generate Timestamp if missing
    if "Timestamp" not in df.columns or df["Timestamp"].isna().all():
        df["Timestamp"] = pd.to_datetime(
            range(len(df)), unit="s", origin="2017-07-01"
        )
        print(f"[loader] Generated synthetic Timestamp column ({len(df)} rows)")
    else:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
        df = df.sort_values("Timestamp").reset_index(drop=True)

    # Generate synthetic src_ip from Label (for campaign grouping)
    # CIC-IDS2017 has no per-flow src_ip — each Label maps to one IP
    if "src_ip" not in df.columns:
        label_to_ip = {}
        for idx, label in enumerate(df["Label"].unique()):
            label_to_ip[str(label).strip()] = f"10.0.{idx // 256}.{idx % 256}"
        df["src_ip"] = df["Label"].map(label_to_ip).fillna("0.0.0.0")

    # Apply row limit if specified
    if max_rows is not None and len(df) > max_rows:
        print(f"[loader] Limiting to {max_rows} rows (was {len(df)})")
        df = df.sample(max_rows, random_state=42).reset_index(drop=True)

    print(f"[loader] CIC-IDS2017 loaded: shape={df.shape}")
    print(f"[loader] Label distribution:\n{df['Label'].value_counts()}")

    return df


def load_unswnb15(data_dir: str) -> pd.DataFrame:
    """Load all UNSW-NB15 CSV files with memory-efficient loading."""
    csv_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    dfs = []
    for f in csv_files:
        df = pd.read_csv(f, low_memory=False)
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    df.columns = df.columns.str.strip()

    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()

    print(f"[loader] UNSW-NB15 loaded: shape={df.shape}")
    label_col = [c for c in df.columns if "label" in c.lower() or "attack" in c.lower()]
    if label_col:
        print(f"[loader] Label column '{label_col[0]}' distribution:")
        print(df[label_col[0]].value_counts())

    return df


def merge_datasets(df1: pd.DataFrame, df2: pd.DataFrame,
                   source_name_1: str = "CIC-IDS2017",
                   source_name_2: str = "UNSW-NB15") -> pd.DataFrame:
    """Merge two datasets with a dataset_source column."""
    df1 = df1.copy()
    df2 = df2.copy()

    df1["dataset_source"] = source_name_1
    df2["dataset_source"] = source_name_2

    merged = pd.concat([df1, df2], ignore_index=True)
    print(f"[loader] Merged dataset: shape={merged.shape}")
    print(f"[loader] Sources: {merged['dataset_source'].value_counts().to_dict()}")

    return merged


if __name__ == "__main__":
    print("=== loader.py self-test ===")
    cicids_dir = "data/raw/cicids2017"
    if os.path.exists(cicids_dir):
        df = load_cicids2017(cicids_dir)
        print(f"Loaded {len(df)} rows from CIC-IDS2017")
    else:
        print(f"Directory {cicids_dir} not found -- skipping")
