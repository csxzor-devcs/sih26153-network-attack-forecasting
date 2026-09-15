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
    handle_nulls, normalise, RobustScaler
from src.pipeline.sequencer import reconstruct_campaigns, create_sliding_windows, \
    create_campaign_splits, save_sequences, print_campaign_stats, _count_windows


def run_pipeline(data_dir="data/raw/cicids2017",
                 output_dir="data/sequences",
                 n_campaigns=100,
                 use_unsw=False,
                 unswnb15_dir="data/raw/unswnb15"):
    """
    Execute the complete data pipeline from raw CSVs to saved sequences.
    Uses the FULL dataset (no subsampling).

    P0 FIX: Scaler is fitted on training data only (no preprocessing leakage).
    Full pipeline flow:
    1. Load & label raw data
    2. Select features & handle infinities/nulls (unsupervised, safe)
    3. Create campaigns & splits FIRST
    4. Fit scaler on train campaign flows ONLY
    5. Transform train/val/test with the same scaler
    6. Create sliding windows & save sequences

    Args:
        data_dir: Path to CIC-IDS2017 CSV files.
        output_dir: Path to save processed sequences.
        n_campaigns: Number of campaigns to create.
        use_unsw: Whether to also load UNSW-NB15.
        unswnb15_dir: Path to UNSW-NB15 CSV files.
    """
    print("=" * 80)
    print("SIH26153 - Data Pipeline Starting")
    print("=" * 80)

    # Step 1: Load CIC-IDS2017 (FULL dataset — no subsampling)
    print("\n[Phase 1/5] Loading CIC-IDS2017 (full dataset)...")
    df = load_cicids2017(data_dir)

    # Step 2: Apply stage labels
    print("\n[Phase 2/5] Applying attack stage labels...")
    print_label_mapping_table(STAGE_MAP, MITRE_MAP)
    df = apply_stage_labels(df, STAGE_MAP)
    df = apply_mitre_labels(df, MITRE_MAP)

    stage_dist = get_stage_distribution(df)
    print(f"\nStage distribution:\n{json.dumps(stage_dist, indent=2)}")

    # Step 3: Feature selection (unsupervised — safe before split)
    print("\n[Phase 3/5] Selecting features...")
    df_features = select_features(df, FEATURE_COLS)

    # Drop unnecessary columns to free memory before heavy ops
    keep_cols = set(FEATURE_COLS + ["Timestamp", "stage", "Label"])
    drop_cols = [c for c in df_features.columns if c not in keep_cols]
    if drop_cols:
        df_features = df_features.drop(columns=drop_cols)

    df_features = handle_infinities(df_features)
    df_features = handle_nulls(df_features)

    # Step 4: Reconstruct campaigns BEFORE fitting scaler
    # P0 FIX: Campaigns are created from raw (non-scaled) features
    # so the scaler can be fitted on train data only
    # No subsampling — full dataset for training
    print("\n[Phase 4/5] Reconstructing attack campaigns (full dataset)...")
    campaigns = reconstruct_campaigns(df_features, subsample=None,
                                       n_campaigns=n_campaigns)
    campaign_stats = print_campaign_stats(campaigns)

    # Step 5: Create campaign splits (before any scaling)
    print("\n[Phase 5a/5] Splitting campaigns into train/val/test...")
    campaign_splits = create_campaign_splits(campaigns, train_ratio=0.7,
                                              val_ratio=0.15)

    # Step 6: Fit scaler on TRAINING campaign data ONLY
    # P0 FIX: No preprocessing leakage — scaler sees only train data
    print("\n[Phase 5b/5] Fitting scaler on training data only...")
    train_campaign_ids = campaign_splits["train"]
    train_flows = []
    for cid in train_campaign_ids:
        if cid in campaigns:
            train_flows.extend(campaigns[cid])

    # Build feature matrix from train flows
    train_feature_matrix = np.array(
        [[flow.get(col, 0.0) for col in FEATURE_COLS]
         for flow in train_flows],
        dtype=np.float32
    )
    print(f"[pipeline] Fitting RobustScaler on {len(train_flows)} train flows...")
    scaler = RobustScaler()
    scaler.fit(train_feature_matrix)
    print(f"[pipeline] Scaler fitted. Center shape: {scaler.center_.shape}")

    # Step 7: Normalise ALL campaigns using train-fitted scaler
    print("\n[Phase 5c/5] Normalising campaigns with train-fitted scaler...")
    for cid, flows in campaigns.items():
        for flow in flows:
            feature_vals = np.array(
                [flow.get(col, 0.0) for col in FEATURE_COLS],
                dtype=np.float32
            ).reshape(1, -1)
            scaled = scaler.transform(feature_vals)[0]
            for j, col in enumerate(FEATURE_COLS):
                flow[col] = float(scaled[j])

    # Save the fitted scaler for inference-time use
    import pickle
    scaler_path = os.path.join("data", "processed", "scaler.pkl")
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"[pipeline] Scaler saved to {scaler_path}")

    # Step 8: Create sliding windows (temporally-safe)
    print("\n[Phase 6/5] Creating sliding-window sequences...")

    # Count total windows for pre-allocation (without materializing)
    total_windows = _count_windows(campaigns, window_size=20, forecast_horizon=1)
    print(f"[pipeline] Total windows to create: {total_windows}")

    # Stream windows directly to disk using memmap (memory-efficient)
    windows = create_sliding_windows(campaigns, window_size=20,
                                      forecast_horizon=1)
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
    print(f"Scaler fitted on TRAIN ONLY (no leakage)")
    print(f"Forecast lead time: {metadata['forecast_lead_time']} flow(s) "
          f"(predict {metadata['forecast_horizon']} step(s) ahead)")
    print(f"Sequences saved to: {output_dir}")
    print("=" * 80)

    return {
        "num_sequences": metadata["num_sequences"],
        "num_campaigns": campaign_stats["total_campaigns"],
        "avg_campaign_length": campaign_stats["avg_campaign_length"],
        "stage_distribution": stage_dist,
        "scaler_path": scaler_path,
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
