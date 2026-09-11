"""
run_pipeline.py -- End-to-end data pipeline for SIH26153.

Memory-optimised pipeline that loads CIC-IDS2017, applies stage and MITRE
labels, selects and normalises features, reconstructs attack campaigns,
creates sliding-window sequences, and saves them for model training.

Usage:
    python -m src.pipeline.run_pipeline
    or
    python src/pipeline/run_pipeline.py
"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
import numpy as np

from src.config import STAGE_MAP, MITRE_MAP, FEATURE_COLS, MODEL_DIR
from src.pipeline.loader import load_cicids2017, load_unswnb15, merge_datasets
from src.pipeline.labeller import apply_stage_labels, apply_mitre_labels, \
    get_stage_distribution, print_label_mapping_table
from src.pipeline.features import select_features, handle_infinities, \
    handle_nulls, normalise
from src.pipeline.sequencer import reconstruct_campaigns, create_sliding_windows, \
    create_campaign_splits, save_sequences, print_campaign_stats


def run_pipeline(data_dir="data/raw/cicids2017",
                 output_dir="data/sequences",
                 max_rows=100000,
                 n_campaigns=100,
                 use_unsw=False,
                 unswnb15_dir="data/raw/unswnb15"):
    """
    Execute the complete data pipeline from raw CSVs to saved sequences.

    Args:
        data_dir: Path to CIC-IDS2017 CSV files.
        output_dir: Path to save processed sequences.
        max_rows: Maximum rows to load to prevent OOM.
        use_unsw: Whether to also load UNSW-NB15.
        unswnb15_dir: Path to UNSW-NB15 CSV files.
    """
    print("=" * 80)
    print("SIH26153 - Data Pipeline Starting")
    print("=" * 80)

    # Step 1: Load CIC-IDS2017 (with row limit to prevent OOM)
    print("\n[Phase 1/5] Loading CIC-IDS2017...")
    df = load_cicids2017(data_dir, max_rows=max_rows)

    # Step 2: Apply stage labels
    print("\n[Phase 2/5] Applying attack stage labels...")
    print_label_mapping_table(STAGE_MAP, MITRE_MAP)
    df = apply_stage_labels(df, STAGE_MAP)
    df = apply_mitre_labels(df, MITRE_MAP)

    stage_dist = get_stage_distribution(df)
    print(f"\nStage distribution:\n{json.dumps(stage_dist, indent=2)}")

    # Step 3: Feature selection + normalisation
    print("\n[Phase 3/5] Selecting and normalising features...")
    df_features = select_features(df, FEATURE_COLS)

    # Drop unnecessary columns to free memory before heavy ops
    # Keep: features, Timestamp, stage, Label (needed for campaigns)
    keep_cols = set(FEATURE_COLS + ["Timestamp", "stage", "Label"])
    drop_cols = [c for c in df_features.columns if c not in keep_cols]
    if drop_cols:
        df_features = df_features.drop(columns=drop_cols)

    df_features = handle_infinities(df_features)
    df_features = handle_nulls(df_features)
    df_normalised, scaler = normalise(df_features)

    # Step 4: Reconstruct campaigns (vectorised, with subsample)
    print("\n[Phase 4/5] Reconstructing attack campaigns...")
    campaigns = reconstruct_campaigns(df_normalised, subsample=max_rows, n_campaigns=n_campaigns)
    campaign_stats = print_campaign_stats(campaigns)

    # Step 5: Create campaign splits
    print("\n[Phase 5a/5] Splitting campaigns into train/val/test...")
    campaign_splits = create_campaign_splits(campaigns, train_ratio=0.7, val_ratio=0.15)

    # Step 6: Create sliding windows (temporally-safe)
    print("\n[Phase 5b/5] Creating sliding-window sequences...")
    windows = create_sliding_windows(campaigns, window_size=20, forecast_horizon=1)
    metadata = save_sequences(windows, output_dir,
                               campaign_splits=campaign_splits,
                               campaigns=campaigns)

    # Summary
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE - SUMMARY")
    print("=" * 80)
    print(f"Created {metadata['num_sequences']} sequences from "
          f"{campaign_stats['total_campaigns']} campaigns")
    print(f"Average campaign length: {campaign_stats['avg_campaign_length']} flows")
    print(f"Sequence shape: {metadata['feature_shape']}")
    print(f"Stage-to-index mapping: {metadata['stage_to_idx']}")
    print(f"Campaign splits: train={len(campaign_splits['train'])}, "
          f"val={len(campaign_splits['val'])}, "
          f"test={len(campaign_splits['test'])}")
    print(f"Sequences saved to: {output_dir}")
    print("=" * 80)

    return {
        "num_sequences": metadata["num_sequences"],
        "num_campaigns": campaign_stats["total_campaigns"],
        "avg_campaign_length": campaign_stats["avg_campaign_length"],
        "stage_distribution": stage_dist,
        "scaler_path": "data/processed/scaler.pkl",
    }


if __name__ == "__main__":
    result = run_pipeline()

    # Verify checkpoint
    print("\n[VERIFICATION]")
    assert os.path.exists("data/sequences/X.npy"), "X.npy not found"
    assert os.path.exists("data/sequences/y.npy"), "y.npy not found"
    assert os.path.exists("data/sequences/metadata.json"), "metadata.json not found"

    X = np.load("data/sequences/X.npy")
    y = np.load("data/sequences/y.npy")
    print(f"X.npy shape: {X.shape}")
    print(f"y.npy shape: {y.shape}")
    print(f"metadata.json exists")

    if result["num_sequences"] >= 500:
        print(f"{result['num_sequences']} sequences created (>= 500 threshold)")
    else:
        print(f"WARNING: Only {result['num_sequences']} sequences (< 500 threshold)")
