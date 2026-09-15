"""
sequencer.py -- Campaign reconstruction and sliding-window sequence creation.

Groups network flows into temporally-contiguous segments and creates
strictly-causal sliding windows for next-stage forecasting.

P0 FIXES:
1. Feature preservation: every flow dict now contains all FEATURE_COLS
2. Forecast target: X[i:i+W] → y[i+W+H-1] (NOT last input position)
3. Split generation: uses same create_sliding_windows() for train/val/test

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

FORECAST_HORIZON = 1  # Predict t+H from window ending at t
FORECAST_LEAD_TIME = 1  # Number of flow intervals between window end and target


def reconstruct_campaigns(df: pd.DataFrame,
                           subsample: Optional[int] = None,
                           n_campaigns: int = 100,
                           min_campaign_length: int = WINDOW_SIZE + 1) -> Dict[str, list]:
    """
    Reconstruct attack campaigns from temporally-ordered network flows.

    When subsample is None, uses the full dataset (no subsampling).

    FIXED: Preserves all FEATURE_COLS in each flow dict so that
    _flows_to_matrix() receives actual feature values, not zeros.

    Args:
        df: DataFrame with 'Timestamp', 'stage', 'Label', and FEATURE_COLS.
        subsample: If not None and df has more rows than this, subsample.
                   None means use all rows (full dataset).
        n_campaigns: Number of campaigns to create.
        min_campaign_length: Minimum flows per campaign.

    Returns:
        Dict mapping campaign_id -> list of flow dicts, each containing
        timestamp, stage, src_ip, dst_ip, src_port, dst_port, protocol,
        AND all FEATURE_COLS values.
    """
    # CRITICAL: Sort by Timestamp to ensure temporal ordering
    df_sorted = df.sort_values("Timestamp").reset_index(drop=True)
    n = len(df_sorted)

    # Only subsample if explicitly requested and needed
    if subsample is not None and n > subsample:
        print(f"[sequencer] Dataset has {n} rows -- subsampling to {subsample}")
        # Use contiguous time segments to preserve temporal structure
        # Pick evenly-spaced time windows across the dataset
        stride = n / subsample
        indices = []
        for i in range(subsample):
            idx = int(i * stride)
            if idx < n:
                indices.append(idx)
        df_sorted = df_sorted.iloc[indices].reset_index(drop=True)
        print(f"[sequencer] After temporal subsample: {len(df_sorted)} rows")

    # Split into n_campaigns contiguous time-ordered segments
    # Each campaign is a continuous time window — preserves attack
    # progression patterns
    segment_size = max(1, len(df_sorted) // n_campaigns)
    campaigns = {}

    for campaign_idx in range(n_campaigns):
        start_idx = campaign_idx * segment_size
        if campaign_idx == n_campaigns - 1:
            end_idx = len(df_sorted)
        else:
            end_idx = start_idx + segment_size

        chunk = df_sorted.iloc[start_idx:end_idx]

        if len(chunk) < min_campaign_length:
            continue

        campaign_id = f"campaign_{campaign_idx}"
        flows = []
        for _, row in chunk.iterrows():
            # P0 FIX: Include ALL feature columns in each flow dict
            flow = {
                "timestamp": str(row["Timestamp"]),
                "stage": row["stage"],
                "src_ip": campaign_id,
                "dst_ip": "10.0.0.1",
                "src_port": int(row.get("Destination Port", 0)) if "Destination Port" in row else 0,
                "dst_port": int(row.get("Source Port", 0)) if "Source Port" in row else 0,
                "protocol": str(row.get("Protocol", "")) if "Protocol" in row else "",
            }
            # P0 FIX: Copy all feature columns
            for col in FEATURE_COLS:
                if col in row.index and pd.notna(row[col]):
                    try:
                        flow[col] = float(row[col])
                    except (ValueError, TypeError):
                        flow[col] = 0.0
                else:
                    flow[col] = 0.0
            flows.append(flow)
        campaigns[campaign_id] = flows

    # Filter: keep only campaigns with 2+ distinct stages and enough length
    multi_stage = {cid: flows for cid, flows in campaigns.items()
                   if len(set(f["stage"] for f in flows)) >= 2
                   and len(flows) >= min_campaign_length}

    n_filtered = len(campaigns) - len(multi_stage)
    print(f"[sequencer] {len(campaigns)} total campaigns, "
          f"{len(multi_stage)} valid multi-stage campaigns "
          f"({n_filtered} filtered out)")

    # Verify feature preservation
    sample_flows = list(multi_stage.values())[0]
    sample_features = [f[col] for f in sample_flows[:3] for col in FEATURE_COLS[:3]]
    nonzero_features = sum(1 for v in sample_features if v != 0.0)
    print(f"[sequencer] Feature preservation check: "
          f"{nonzero_features}/{len(sample_features)} sample features are nonzero")

    return multi_stage


def create_sliding_windows(campaigns: Dict[str, list],
                           window_size: int = WINDOW_SIZE,
                           forecast_horizon: int = FORECAST_HORIZON):
    """
    Create strictly-causal sliding-window sequences.

    FIXED: Window [i:i+W] predicts stage at position i+W+forecast_horizon-1.
    This is the true forecasting target, NOT the last input position.

    For forecast_horizon=1:
        X = flows[i], flows[i+1], ..., flows[i+19]   (20 flows)
        y = stage at position i+20                     (the 21st flow = next stage)

    Args:
        campaigns: Dict mapping campaign_id -> list of flow dicts.
        window_size: Number of flows in each input window.
        forecast_horizon: How many steps ahead to predict.

    Yields:
        (X, y) tuples where X shape=(window_size, n_features),
        y is integer stage label at position i+window_size+forecast_horizon-1.
    """
    count = 0
    n_skipped = 0

    for src_ip, flows in campaigns.items():
        if len(flows) < window_size + forecast_horizon:
            continue

        stages = [f["stage"] for f in flows]

        for i in range(len(flows) - window_size - forecast_horizon + 1):
            window_flows = flows[i:i + window_size]
            target_idx = i + window_size + forecast_horizon - 1
            next_stage = stages[target_idx]

            y = STAGE_TO_IDX.get(next_stage, -1)
            if y == -1:
                continue

            try:
                X = _flows_to_matrix(window_flows)
                yield (X, y)
                count += 1
            except (KeyError, ValueError):
                n_skipped += 1
                continue

    print(f"[sequencer] Created {count} sequences "
          f"(window_size={window_size}, forecast_horizon={forecast_horizon}, "
          f"skipped={n_skipped})")


def create_campaign_splits(campaigns: Dict[str, list],
                           train_ratio: float = 0.7,
                           val_ratio: float = 0.15) -> Dict[str, List[str]]:
    """Split campaigns into train/val/test by campaign ID (no row-level leakage)."""
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


def _count_windows(campaigns: Dict[str, list],
                   window_size: int = WINDOW_SIZE,
                   forecast_horizon: int = FORECAST_HORIZON) -> int:
    """Count total windows without materializing them (for pre-allocation)."""
    count = 0
    for flows in campaigns.values():
        if len(flows) < window_size + forecast_horizon:
            continue
        stages = [f["stage"] for f in flows]
        for i in range(len(flows) - window_size - forecast_horizon + 1):
            target_idx = i + window_size + forecast_horizon - 1
            next_stage = stages[target_idx]
            if STAGE_TO_IDX.get(next_stage, -1) != -1:
                count += 1
    return count


def _save_split_streaming(split_campaigns: Dict[str, list],
                          output_dir: str,
                          split_name: str,
                          window_size: int = WINDOW_SIZE,
                          forecast_horizon: int = FORECAST_HORIZON) -> int:
    """Save a campaign split using streaming memmap to avoid holding all in memory.

    Returns the number of sequences saved.
    """
    n_seqs = _count_windows(split_campaigns, window_size, forecast_horizon)
    if n_seqs == 0:
        return 0

    n_features = len(FEATURE_COLS)
    X_path = os.path.join(output_dir, f"X_{split_name}.npy")
    y_path = os.path.join(output_dir, f"y_{split_name}.npy")

    # Use raw temp files for memmap streaming, then convert to proper .npy
    X_tmp = X_path + ".tmp"
    y_tmp = y_path + ".tmp"

    # Stream windows into raw memmap (no .npy header)
    X_mm = np.memmap(X_tmp, dtype=np.float32, mode='w+',
                      shape=(n_seqs, window_size, n_features))
    y_mm = np.memmap(y_tmp, dtype=np.int64, mode='w+',
                      shape=(n_seqs,))

    idx = 0
    for X, y in create_sliding_windows(split_campaigns, window_size, forecast_horizon):
        X_mm[idx] = X
        y_mm[idx] = y
        idx += 1

    X_mm.flush()
    y_mm.flush()
    del X_mm, y_mm

    # Convert to proper .npy by reading raw data and saving with header
    # Use copy=False to avoid duplicating the large array in memory
    X_raw = np.memmap(X_tmp, dtype=np.float32, mode='r',
                       shape=(n_seqs, window_size, n_features))
    y_raw = np.memmap(y_tmp, dtype=np.int64, mode='r', shape=(n_seqs,))

    # np.save reads from memmap incrementally, writes proper .npy
    np.save(X_path, X_raw[:idx])
    np.save(y_path, y_raw[:idx])

    del X_raw, y_raw
    os.remove(X_tmp)
    os.remove(y_tmp)

    print(f"[sequencer] Saved {idx} {split_name} sequences "
          f"with H={forecast_horizon} forecast target")
    return idx


def save_sequences(windows,
                   output_dir: str = SEQUENCE_DIR,
                   campaign_splits: Optional[Dict[str, List[str]]] = None,
                   campaigns: Optional[Dict[str, list]] = None) -> dict:
    """Save sequences as X.npy, y.npy, metadata.json, and campaign split files.

    Accepts a generator (from create_sliding_windows) for memory-efficient
    streaming writes using pre-allocated numpy memmap.

    P0 FIX: Split datasets use create_sliding_windows() to ensure
    forecast targets are consistent (X[i:i+W] → y[i+W+H-1]).
    """
    os.makedirs(output_dir, exist_ok=True)

    n_features = len(FEATURE_COLS)
    window_size = WINDOW_SIZE
    forecast_horizon = FORECAST_HORIZON

    # Count total windows for pre-allocation
    print("[sequencer] Counting total windows for pre-allocation...")
    total_count = 0
    # We need to count windows across ALL campaigns (since the generator
    # iterates over everything). We count per-split by splitting campaigns first.
    if campaign_splits and campaigns:
        total_count = sum(
            _count_windows({cid: campaigns[cid] for cid in cids if cid in campaigns},
                          window_size, forecast_horizon)
            for cids in campaign_splits.values()
        )
    else:
        total_count = _count_windows(campaigns or {}, window_size, forecast_horizon)

    print(f"[sequencer] Pre-allocating for {total_count} sequences "
          f"(shape: ({total_count}, {window_size}, {n_features}))")

    # Pre-allocate memmap arrays (use temp files, convert to .npy after)
    X_path = os.path.join(output_dir, "X.npy")
    y_path = os.path.join(output_dir, "y.npy")
    X_tmp = X_path + ".tmp"
    y_tmp = y_path + ".tmp"

    X_mm = np.memmap(X_tmp, dtype=np.float32, mode='w+',
                     shape=(total_count, window_size, n_features))
    y_mm = np.memmap(y_tmp, dtype=np.int64, mode='w+', shape=(total_count,))

    # Stream windows into memmap
    idx = 0
    for X, y in windows:
        X_mm[idx] = X
        y_mm[idx] = y
        idx += 1
        if idx % 100000 == 0:
            print(f"[sequencer] Streamed {idx}/{total_count} sequences...")

    X_mm.flush()
    y_mm.flush()
    del X_mm, y_mm

    # Convert to proper .npy format
    X_raw = np.memmap(X_tmp, dtype=np.float32, mode='r',
                       shape=(total_count, window_size, n_features))
    y_raw = np.memmap(y_tmp, dtype=np.int64, mode='r', shape=(total_count,))
    np.save(X_path, X_raw[:idx])
    np.save(y_path, y_raw[:idx])
    del X_raw, y_raw
    os.remove(X_tmp)
    os.remove(y_tmp)

    # Verify
    X_shape = (idx, window_size, n_features)
    print(f"[sequencer] Saved {idx} sequences to {output_dir}")
    print(f"[sequencer] X.npy shape: {X_shape}, y.npy shape: ({idx},)")

    metadata = {
        "num_sequences": idx,
        "window_size": window_size,
        "forecast_horizon": forecast_horizon,
        "forecast_lead_time": FORECAST_LEAD_TIME,
        "feature_shape": list(X_shape[1:]),
        "stage_to_idx": STAGE_TO_IDX,
        "idx_to_stage": {v: k for k, v in STAGE_TO_IDX.items()},
        "description": "Strictly-causal sliding-window sequences for attack forecasting",
        "campaign_splits": campaign_splits if campaign_splits else {},
        "forecast_target_explanation": (
            "X[i:i+W] predicts stage at position i+W+FORECAST_HORIZON-1. "
            "With FORECAST_HORIZON=1, the model predicts the stage that "
            "begins immediately after the input window ends."
        ),
    }

    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    # Save campaign-level splits using streaming memmap
    if campaign_splits and campaigns:
        for split_name, campaign_ids in campaign_splits.items():
            split_campaigns = {cid: campaigns[cid] for cid in campaign_ids if cid in campaigns}
            if split_campaigns:
                _save_split_streaming(split_campaigns, output_dir, split_name,
                                      window_size, forecast_horizon)

    print(f"[sequencer] All sequences saved to {output_dir}")

    return metadata


def print_campaign_stats(campaigns: Dict[str, list]) -> dict:
    """Print and return campaign statistics including transition matrix."""
    total_campaigns = len(campaigns)
    lengths = [len(flows) for flows in campaigns.values()]
    avg_length = np.mean(lengths) if lengths else 0

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
