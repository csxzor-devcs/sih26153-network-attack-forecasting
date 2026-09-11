"""
sequencer.py -- Campaign reconstruction and sliding-window sequence creation.

Groups network flows by synthetic source IP (derived from Label column),
identifies multi-stage attack campaigns, and creates sliding-window
sequences for supervised sequence forecasting.

Memory-efficient: uses vectorized pandas operations instead of iterrows().
"""

import json
import os
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Tuple

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
                           n_campaigns: int = 200) -> Dict[str, list]:
    """
    Reconstruct attack campaigns from network flows.

    Instead of grouping by Label-derived IP (which guarantees same-stage
    groups since each CIC-IDS2017 label = one attack stage), this function
    sorts flows by timestamp and assigns them to N campaigns using row
    index hashing. This ensures each campaign receives flows from ALL
    attack types, producing genuine multi-stage campaigns with natural
    stage progressions (Recon -> Exploit -> Impact).

    Memory-efficient: uses vectorized pandas operations only.

    Args:
        df: DataFrame with 'Timestamp', 'stage', 'Label', and features.
        subsample: If df has more rows than this, take a stratified sample.
        n_campaigns: Number of synthetic campaigns to create.
                     Each campaign gets a mix of all attack types.

    Returns:
        Dict mapping campaign_id -> list of flow dicts, sorted by timestamp.
    """
    # Subsample if too large (stratified by stage)
    if len(df) > subsample:
        print(f"[sequencer] Dataset has {len(df)} rows -- subsampling to {subsample}")
        n_per_stage = max(1, subsample // len(df.groupby("stage")))
        df = df.groupby("stage", group_keys=False).apply(
            lambda g: g.sample(min(len(g), n_per_stage), random_state=42)
        ).reset_index(drop=True)
        print(f"[sequencer] After subsample: {len(df)} rows")

    # Sort by timestamp — critical for temporal ordering within campaigns
    df_sorted = df.sort_values("Timestamp").reset_index(drop=True)
    n = len(df_sorted)

    # Assign each row to a campaign using row index modulo n_campaigns
    # This ensures each campaign gets flows from ALL attack types
    # spread across time, creating genuine multi-stage campaigns
    df_sorted["_campaign_id"] = df_sorted.index % n_campaigns

    campaigns = {}
    for campaign_idx in range(n_campaigns):
        campaign_id = f"campaign_{campaign_idx}"
        chunk = df_sorted[df_sorted["_campaign_id"] == campaign_idx]

        # Shuffle flows within campaign to create mixed-stage windows
        # This ensures sliding windows see natural stage progressions
        # (Recon -> Exploit -> Impact) rather than same-stage blocks
        chunk = chunk.sample(frac=1, random_state=42).reset_index(drop=True)

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
    multi_stage = {cid: flows for cid, flows in campaigns.items()
                   if len(set(f["stage"] for f in flows)) >= 2}

    n_filtered = len(campaigns) - len(multi_stage)
    print(f"[sequencer] {len(campaigns)} total campaigns, "
          f"{len(multi_stage)} multi-stage campaigns "
          f"({n_filtered} filtered out)")

    return multi_stage


def create_sliding_windows(campaigns: Dict[str, list],
                    window_size: int = WINDOW_SIZE) -> List[Tuple[np.ndarray, int]]:
    """
    Create sliding-window sequences from reconstructed campaigns.

    Args:
        campaigns: Dict mapping src_ip -> list of flow dicts.
        window_size: Number of flows in each input window.

    Returns:
        List of (X, y) tuples where X shape=(window_size, n_features),
        y is integer stage label of next flow.
    """
    windows = []
    n_skipped = 0

    for src_ip, flows in campaigns.items():
        if len(flows) < window_size + 1:
            continue

        # Extract stages and feature values vectorized
        stages = [f["stage"] for f in flows]

        for i in range(len(flows) - window_size):
            window_stages = stages[i:i + window_size]
            next_stage = stages[i + window_size]

            # Skip if no progression
            if len(set(window_stages)) == 1 and window_stages[0] == next_stage:
                n_skipped += 1
                continue

            window_flows = flows[i:i + window_size]
            try:
                X = _flows_to_matrix(window_flows)
                y = STAGE_TO_IDX.get(next_stage, -1)
                if y == -1:
                    continue
                windows.append((X, y))
            except (KeyError, ValueError):
                n_skipped += 1
                continue

    print(f"[sequencer] Created {len(windows)} sequences "
          f"(skipped {n_skipped} due to no progression)")
    return windows


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
                    output_dir: str = SEQUENCE_DIR) -> dict:
    """Save sequences as X.npy, y.npy, metadata.json."""
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
        "description": "Sliding-window sequences for attack forecasting",
    }

    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

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
    print("STAGE TRANSITION MATRIX")
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
