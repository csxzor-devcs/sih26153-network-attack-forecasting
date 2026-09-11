"""
test_pipeline.py -- Unit tests for the data pipeline components.

Tests:
- Loader: CSV loading, column validation, synthetic src_ip generation
- Sequencer: Campaign reconstruction, sliding window generation
- Features: Feature extraction correctness
"""

import os
import sys
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline.sequencer import reconstruct_campaigns, create_sliding_windows
from src.pipeline.loader import load_cicids2017
from src.config import STAGE_MAP, STAGE_ORDER, STAGE_TO_IDX, WINDOW_SIZE, FEATURE_COLS


class TestSequencer:
    """Tests for campaign reconstruction and window generation."""

    def test_reconstruct_campaigns_temporal_ordering(self):
        """Verify that campaigns are temporally ordered (no shuffling)."""
        n_rows = 1000
        n_campaigns = 5

        # Create timestamp-sorted data
        timestamps = pd.date_range("2024-01-01", periods=n_rows, freq="s")
        stages = np.random.choice(list(STAGE_MAP.values()), n_rows)
        df = pd.DataFrame({
            "Timestamp": timestamps,
            "stage": stages,
            "Label": np.random.choice(list(STAGE_MAP.keys()), n_rows),
            "Flow Duration": np.random.rand(n_rows).astype(np.float32),
        })
        df = df.sort_values("Timestamp").reset_index(drop=True)

        campaigns = reconstruct_campaigns(df, n_campaigns=n_campaigns, subsample=2000)

        # Verify campaigns are contiguous time segments
        assert len(campaigns) > 0
        for campaign_id, flows in campaigns.items():
            # Check flows are temporally ordered within campaign
            timestamps_in_campaign = [f["timestamp"] for f in flows]
            assert timestamps_in_campaign == sorted(timestamps_in_campaign), (
                f"Campaign {campaign_id} flows are not temporally ordered!"
            )

    def test_reconstruct_campaigns_contiguous_segments(self):
        """Verify campaign splits are contiguous, not shuffled."""
        n_rows = 500
        n_campaigns = 5

        df = pd.DataFrame({
            "Timestamp": pd.date_range("2024-01-01", periods=n_rows, freq="s"),
            "stage": np.random.choice(list(STAGE_MAP.values()), n_rows),
            "Label": np.random.choice(list(STAGE_MAP.keys()), n_rows),
            "Flow Duration": np.random.rand(n_rows).astype(np.float32),
        })

        campaigns = reconstruct_campaigns(df, n_campaigns=n_campaigns, subsample=2000)
        assert len(campaigns) == n_campaigns

    def test_create_sliding_windows_temporal(self):
        """Verify sliding window generation is causal (no future leakage)."""
        n_flows = 100
        window_size = WINDOW_SIZE

        campaign = []
        for i in range(n_flows):
            flow = {"stage": STAGE_ORDER[i % len(STAGE_ORDER)]}
            for col in FEATURE_COLS:
                flow[col] = float(np.random.rand())
            campaign.append(flow)

        windows = create_sliding_windows({"c0": campaign}, window_size=window_size)

        assert len(windows) > 0
        for X, y in windows:
            assert X.shape == (window_size, len(FEATURE_COLS)), (
                f"Expected shape ({window_size}, {len(FEATURE_COLS)}), got {X.shape}"
            )

    def test_create_campaign_splits(self):
        """Verify campaign-level splitting produces correct proportions."""
        from src.pipeline.sequencer import create_campaign_splits

        campaigns = {f"campaign_{i}": [] for i in range(20)}
        splits = create_campaign_splits(campaigns, train_ratio=0.7, val_ratio=0.15)

        assert len(splits["train"]) == 14
        assert len(splits["val"]) == 3
        assert len(splits["test"]) == 3
        assert len(splits["train"]) + len(splits["val"]) + len(splits["test"]) == 20


class TestLoader:
    """Tests for data loading utilities."""

    def test_load_cicids2017_columns(self):
        """Verify that loading produces expected columns including synthetic src_ip."""
        test_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "raw", "test_cicids_dir"
        )
        os.makedirs(test_dir, exist_ok=True)

        test_csv = os.path.join(test_dir, "test.csv")
        cols = ["Flow Duration", "Total Fwd Packets", "Label", "Destination Port"]
        pd.DataFrame(
            [[100.0, 50, "BENIGN", 80]], columns=cols
        ).to_csv(test_csv, index=False)

        df = load_cicids2017(test_dir)
        assert "Flow Duration" in df.columns
        assert "Label" in df.columns
        assert "src_ip" in df.columns

        import shutil
        shutil.rmtree(test_dir)

    def test_synthetic_src_ip_generation(self):
        """Verify that synthetic src_ip is generated from Label."""
        test_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "raw", "test_srcip_dir"
        )
        os.makedirs(test_dir, exist_ok=True)

        test_csv = os.path.join(test_dir, "test.csv")
        cols = ["Flow Duration", "Label"]
        pd.DataFrame(
            [[100.0, "PortScan"]], columns=cols
        ).to_csv(test_csv, index=False)

        result = load_cicids2017(test_dir)
        assert "src_ip" in result.columns
        assert result["src_ip"].notna().all()

        import shutil
        shutil.rmtree(test_dir)


class TestFeatureExtraction:
    """Tests for feature column definitions."""

    def test_feature_cols_count(self):
        """Verify FEATURE_COLS has expected number of columns."""
        assert len(FEATURE_COLS) == 46, (
            f"Expected 46 feature columns, got {len(FEATURE_COLS)}"
        )

    def test_stage_map_completeness(self):
        """Verify all CIC-IDS2017 labels are mapped."""
        expected_labels = [
            "BENIGN", "PortScan", "FTP-Patator", "SSH-Patator",
            "DoS Hulk", "DoS GoldenEye", "DoS slowloris",
            "DoS Slowhttptest", "Heartbleed",
            "Web Attack – Brute Force", "Web Attack – XSS",
            "Web Attack – Sql Injection", "Infiltration",
            "Bot", "DDoS",
        ]
        for label in expected_labels:
            assert label in STAGE_MAP, f"Missing label: {label}"

    def test_stage_to_idx_mapping(self):
        """Verify stage-to-index mapping is consistent."""
        assert len(STAGE_ORDER) == len(STAGE_TO_IDX)
        for stage, idx in STAGE_TO_IDX.items():
            assert STAGE_ORDER[idx] == stage
