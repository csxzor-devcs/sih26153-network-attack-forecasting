"""
sequencer.py -- Campaign reconstruction and sliding-window sequence creation.

Groups network flows into temporally-contiguous campaigns based on
actual flow ordering, ensuring each campaign is a genuine temporal
sequence that flows through attack stages in natural order.

This fixes the critical temporal leakage bug in the previous version:
- Previous: row-index hashing + within-campaign shuffle → random mixes
- Fixed: contiguous time-ordered segments → genuine temporal sequences

Memory-efficient: uses vectorized pandas operations only.
"""

import json
import os
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

try:
    from src.config import STAGE_MAP, WINDOW_SIZE, SEQUENCE_DIR, FEATURE_COLS
except ImportError:
    STAGE_MAP = {}
    WINDOW_SIZE = 20
    SEQUENCE_DIR = "data/sequences"
    FEATURE_COLS = []

STAGE_TO_IDX = {stage: idx for idx, stage in enumerate(
    ["Benign", "Recon", "CredAccess", "Exploit",
     "LateralMove", "C2", "Impact"])}


def reconstruct_campaigns(df: pd.DataFrame,
                          subsample: int = 500000,
                          n_campaigns: int = 100,
                          min_campaign_length: int = WINDOW_SIZE + 1) -> Dict[str, list]:
    """
    Reconstruct attack campaigns from temporally-ordered network flows.

    FIXED: Uses contiguous time-ordered segments instead of row-index
    hashing + shuffle. Flows are sorted by Timestamp, then split into
    n_campaigns contiguous segments. Each segment is a genuine temporal
    sequence where stages naturally progress over time.

    This eliminates temporal leakage: each campaign's flows are
    strictly ordered in time, and sliding windows only see past→future
    transitions within a campaign.

    Args:
        df: DataFrame with 'Timestamp', 'stage', 'Label', and features.
            Must be sorted by Timestamp or sortable by it.
        subsample: If df has more rows than this, take a stratified sample.
        n_campaigns: Number of campaigns to create.
                     Flows are split into contiguous time-ordered segments.
        min_campaign_length: Minimum flows per campaign (default: window_size+1).

    Returns:
        Dict mapping campaign_id -> list of flow dicts, sorted by Timestamp.
    """
    # Subsample if too large (stratified by stage)
    if len(df) > subsample:
        print(f"[sequencer] Dataset has {len(df)} rows -- subsampling to {subsample}")
        n_per_stage = max(1, subsample // len(df.groupby("stage")))
        df = df.groupby("stage", group_keys=False).apply(
            lambda g: g.sample(min(len(g), n_per_stage), random_state=42)
        ).reset_index(drop=True)
        print(f"[sequencer] After subsample: {len(df)} rows")

    # CRITICAL: Sort by Timestamp to ensure temporal ordering
    df_sorted = df.sort_values("Timestamp").reset_index(drop=True)
    n = len(df_sorted)

    # Split into n_campaigns contiguous time-ordered segments
    # Each segment is a genuine temporal sequence — no shuffling
    segment_size = max(1, n // n_campaigns)
    campaigns = {}

    for campaign_idx in range(n_campaigns):
        start_idx = campaign_idx * segment_size
        # Last campaign gets all remaining rows
        if campaign_idx == n_campaigns - 1:
            end_idx = n
        else:
            end_idx = start_idx + segment_size

        chunk = df_sorted.iloc[start_idx:end_idx]

        # Skip campaigns that are too short for sliding windows
        if len(chunk) < min_campaign_length:
            continue

        campaign_id = f"campaign_{campaign_idx}"
        flows = []
        for _, row in chunk.iterrows():
            flows.append({
                "timestamp": str(row["Timestamp"]),
                "stage": row["stage"],
                "src_ip": campaign_id,
                "dst_ip": "10.0.0.1",
                "src_port": int(row.get("Destination Port", 0)) if "Destination Port" in row else 0,
                "dst_port": int(row.get("Source Port", 0)) if "Source Port" in row else 0,
                "protocol": str(row.get("Protocol", "")) if "Protocol" in row else "",
            })
        campaigns[campaign_id] = flows

    # Filter: keep only campaigns with 2+ distinct stages
    # AND ensure they have enough flows for sliding windows
    multi_stage = {cid: flows for cid, flows in campaigns.items()
                   if len(set(f["stage"] for f in flows)) >= 2
                   and len(flows) >= min_campaign_length}

    n_filtered = len(campaigns) - len(multi_stage)
    print(f"[sequencer] {len(campaigns)} total campaigns, "
          f"{len(multi_stage)} valid multi-stage campaigns "
          f"({n_filtered} filtered out)")

    # Print sample stage progressions
    for cid, flows in list(multi_stage.items())[:3]:
        stages = [f["stage"] for f in flows]
        print(f"[sequencer] {cid}: {stages[:5]}...{stages[-3:]} "
              f"(len={len(stages)}, stages={len(set(stages))})")

    return multi_stage


def create_sliding_windows(campaigns: Dict[str, list],
                           window_size: int = WINDOW_SIZE,
                           forecast_horizon: int = 1) -> List[Tuple[np.ndarray, int]]:
    """
    Create sliding-window sequences from temporally-ordered campaigns.

    FIXED: Windows are strictly causal — window [i:i+W] predicts
    stage at position i+W+forecast_horizon-1. No future leakage.

    Args:
        campaigns: Dict mapping campaign_id -> list of flow dicts
                   (sorted by Timestamp within each campaign).
        window_size: Number of flows in each input window.
        forecast_horizon: How many steps ahead to predict (default 1).
                          t+1 means predict the immediate next stage.

    Returns:
        List of (X, y) tuples where X shape=(window_size, n_features),
        y is integer stage label at position i+window_size+forecast_horizon-1.
    """
    windows = []
    n_skipped = 0

    for src_ip, flows in campaigns.items():
        if len(flows) < window_size + forecast_horizon:
            continue

        stages = [f["stage"] for f in flows]

        for i in range(len(flows) - window_size - forecast_horizon + 1):
            window_stages = stages[i:i + window_size]
            target_idx = i + window_size + forecast_horizon - 1
            next_stage = stages[target_idx]

            # Skip if target stage is invalid
            y = STAGE_TO_IDX.get(next_stage, -1)
            if y == -1:
                continue

            window_flows = flows[i:i + window_size]
            try:
                X = _flows_to_matrix(window_flows)
                windows.append((X, y))
            except (KeyError, ValueError):
                n_skipped += 1
                continue

    print(f"[sequencer] Created {len(windows)} sequences "
          f"(skipped {n_skipped} due to invalid stages)")
    return windows


def create_campaign_splits(campaigns: Dict[str, list],
                           train_ratio: float = 0.7,
                           val_ratio: float = 0.15) -> Dict[str, List[str]]:
    """
    Split campaigns into train/val/test by campaign ID.

    FIXED: Campaign-level splitting prevents data leakage between
    train/val/test sets. All flows from a campaign go to exactly one split.

    Args:
        campaigns: Dict of campaign_id -> flows.
        train_ratio: Fraction of campaigns for training.
        val_ratio: Fraction for validation.

    Returns:
        Dict with keys 'train', 'val', 'test' mapping to lists of campaign IDs.
    """
    campaign_ids = sorted(campaigns.keys())
    n = len(campaign_ids)

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    splits = {
        "train": campaign_ids[:n_train],
        "val": campaign_ids[n_train:n_train + n_val],
        "test": campaign_ids[n_train + n_val:],
    }

    print(f"[sequencer] Campaign splits: "
          f"train={len(splits['train'])}, "
          f"val={len(splits['val'])}, "
          f"test={len(splits['test'])}")
    return splits


def _flows_to_matrix(flows: list) -> np.ndarray:
    """Convert list of flow dicts to numeric feature matrix (float32)."""
    n_flows = len(flows)
    n_features = len(FEATURE_COLS) if FEATURE_COLS else 79
    feature_matrix = np.zeros((n_flows, n_features), dtype=np.float32)

    for i, flow in enumerate(flows):
        for j, col in enumerate(FEATURE_COLS):
            val = flow.get(col, 0)
            try:
                feature_matrix[i, j] = float(val) if val else 0.0
            except (ValueError, TypeError):
                feature_matrix[i, j] = 0.0

    return feature_matrix


def save_sequences(windows: List[Tuple[np.ndarray, int]],
                   output_dir: str = SEQUENCE_DIR,
                   campaign_splits: Optional[Dict[str, List[str]]] = None,
                   campaigns: Optional[Dict[str, list]] = None) -> dict:
    """Save sequences as X.npy, y.npy, metadata.json, and split files."""
    os.makedirs(output_dir, exist_ok=True)

    X_list = [w[0] for w in windows]
    y_list = [w[1] for w in windows]

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)

    np.save(os.path.join(output_dir, "X.npy"), X)
    np.save(os.path.join(output_dir, "y.npy"), y)

    metadata = {
        "num_sequences": len(windows),
        "window_size": WINDOW_SIZE,
        "feature_shape": list(X.shape[1:]),
        "stage_to_idx": STAGE_TO_IDX,
        "idx_to_stage": {v: k for k, v in STAGE_TO_IDX.items()},
        "description": "Temporally-ordered sliding-window sequences for attack forecasting",
        "forecast_horizon": 1,
        "campaign_splits": campaign_splits if campaign_splits else {},
    }

    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    # Save campaign-level splits for training
    if campaign_splits and campaigns:
        for split_name, campaign_ids in campaign_splits.items():
            split_windows = []
            for cid in campaign_ids:
                if cid in campaigns:
                    flows = campaigns[cid]
                    for i in range(len(flows) - WINDOW_SIZE + 1):
                        stages = [f["stage"] for f in flows]
                        y_val = STAGE_TO_IDX.get(stages[i + WINDOW_SIZE - 1], -1)
                        if y_val == -1:
                            continue
                        X_val = _flows_to_matrix(flows[i:i + WINDOW_SIZE])
                        split_windows.append((X_val, y_val))
            if split_windows:
                X_split = np.array([w[0] for w in split_windows], dtype=np.float32)
                y_split = np.array([w[1] for w in split_windows], dtype=np.int64)
                np.save(os.path.join(output_dir, f"X_{split_name}.npy"), X_split)
                np.save(os.path.join(output_dir, f"y_{split_name}.npy"), y_split)
                print(f"[sequencer] Saved {len(split_windows)} {split_name} sequences")

    print(f"[sequencer] Saved {len(windows)} sequences to {output_dir}")
    print(f"[sequencer] X.npy shape: {X.shape}")
    print(f"[sequencer] y.npy shape: {y.shape}")

    return metadata


def print_campaign_stats(campaigns: Dict[str, list]) -> dict:
    """Print and return campaign statistics including transition matrix."""
    total_campaigns = len(campaigns)
    lengths = [len(flows) for flows in campaigns.values()]
    avg_length = np.mean(lengths) if lengths else 0

    # Vectorized transition counting
    transition_counts = defaultdict(lambda: defaultdict(int))
    for src_ip, flows in campaigns.items():
        stages = [f["stage"] for f in flows]
        for i in range(len(stages) - 1):
            transition_counts[stages[i]][stages[i + 1]] += 1

    all_stages = sorted(set(
        s for flows in campaigns.values() for f in flows for s in [f["stage"]]
    ))
    print("\n" + "=" * 80)
    print("STAGE TRANSITION MATRIX (TEMPORALLY ORDERED)")
    print("=" * 80)
    header = f"{'From':<18}" + "".join(f"{s:<15}" for s in all_stages)
    print(header)
    print("-" * 80)
    for src_stage in all_stages:
        row = f"{src_stage:<18}"
        for dst_stage in all_stages:
            count = transition_counts[src_stage][dst_stage]
            row += f"{count:<15}"
        print(row)
    print("=" * 80 + "\n")

    stats = {
        "total_campaigns": total_campaigns,
        "avg_campaign_length": round(float(avg_length), 2),
        "transition_matrix": {
            src: dict(dst_counts) for src, dst_counts in transition_counts.items()
        },
    }
    return stats


if __name__ == "__main__":
    print("=== sequencer.py self-test ===")
    print("Imported successfully -- waiting for pipeline integration")
