"""
test_models.py -- Unit tests for model implementations.

Tests:
- Markov baseline: Transition matrix, prediction correctness
- LSTM: Model architecture, forward pass
- Transformer: Attention mechanism, training
"""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.baseline.markov import MarkovBaseline
from src.baseline.comparison import MajorityClassifier, XGBoostBaseline
from src.config import STAGE_ORDER, STAGE_TO_IDX, WINDOW_SIZE, FEATURE_COLS


class TestMarkovBaseline:
    """Tests for the Markov chain baseline."""

    def test_markov_initialization(self):
        """Verify Markov model initializes correctly."""
        model = MarkovBaseline(n_stages=len(STAGE_ORDER))
        assert model.n_stages == len(STAGE_ORDER)
        assert model.transition_probs.shape == (len(STAGE_ORDER), len(STAGE_ORDER))

    def test_fit_with_sequences(self):
        """Verify fit accepts and processes sequence data."""
        model = MarkovBaseline(n_stages=len(STAGE_ORDER))

        n_stages = len(STAGE_ORDER)
        sequences = [np.random.randint(0, n_stages, size=WINDOW_SIZE) for _ in range(50)]
        model.fit(np.array([s for s in sequences]).flatten())

        assert model.trained is True
        assert model.transition_probs.sum(axis=1).mean() > 0

    def test_predict_next_shape(self):
        """Verify predict_next returns correct output."""
        model = MarkovBaseline(n_stages=len(STAGE_ORDER))

        n_stages = len(STAGE_ORDER)
        sequences = [np.random.randint(0, n_stages, size=WINDOW_SIZE) for _ in range(50)]
        model.fit(np.array([s for s in sequences]).flatten())

        stage_idx = np.random.randint(0, n_stages)
        next_stage, prob = model.predict_next(stage_idx)

        assert isinstance(next_stage, (int, np.integer))
        assert 0 <= next_stage < n_stages
        assert 0 <= prob <= 1

    def test_transition_matrix_properties(self):
        """Verify transition probabilities rows sum to 1."""
        model = MarkovBaseline(n_stages=len(STAGE_ORDER))

        n_stages = len(STAGE_ORDER)
        sequences = [np.random.randint(0, n_stages, size=WINDOW_SIZE) for _ in range(200)]
        model.fit(np.array([s for s in sequences]).flatten())

        row_sums = model.transition_probs.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=0.01)


class TestModelArchitecture:
    """Tests for model architecture components."""

    def test_lstm_model_structure(self):
        """Verify LSTM model has correct parameter count."""
        from src.models.lstm import AttackLSTM
        import torch

        model = AttackLSTM(n_features=46, hidden_size=128, n_layers=2, n_stages=7)
        n_params = sum(p.numel() for p in model.parameters())

        # LSTM(46→128, 2 layers, dropout=0.3, FC→7) ≈ 223K params
        assert 200000 < n_params < 250000, (
            f"Expected ~223K params, got {n_params}"
        )

    def test_transformer_model_structure(self):
        """Verify Transformer model has correct parameter count."""
        from src.models.transformer import AttackTransformer
        import torch

        model = AttackTransformer(
            n_features=46, d_model=128, n_heads=4,
            n_layers=4, d_ff=256, n_stages=7
        )
        n_params = sum(p.numel() for p in model.parameters())

        # Transformer(46→128, 4 layers) ≈ 543K params
        assert 500000 < n_params < 600000, (
            f"Expected ~543K params, got {n_params}"
        )

    def test_transformer_forward_shape(self):
        """Verify Transformer forward pass produces correct output."""
        from src.models.transformer import AttackTransformer
        import torch

        model = AttackTransformer(n_features=46, d_model=128, n_heads=4, n_layers=4)
        batch_size = 4
        seq_len = WINDOW_SIZE
        n_features = 46

        x = torch.randn(batch_size, seq_len, n_features)
        output = model(x)

        assert output.shape == (batch_size, 7)

    def test_lstm_forward_shape(self):
        """Verify LSTM forward pass produces correct output."""
        from src.models.lstm import AttackLSTM
        import torch

        model = AttackLSTM(n_features=46, hidden_size=128, n_layers=2, n_stages=7)
        batch_size = 4
        seq_len = WINDOW_SIZE
        n_features = 46

        x = torch.randn(batch_size, seq_len, n_features)
        output = model(x)

        assert output.shape == (batch_size, 7)


class TestModelTraining:
    """Tests for model training behavior."""

    def test_markov_train_accuracy(self):
        """Verify Markov model achieves reasonable accuracy."""
        model = MarkovBaseline(n_stages=len(STAGE_ORDER))

        n_stages = len(STAGE_ORDER)
        n_sequences = 200
        sequences = [np.random.randint(0, n_stages, size=WINDOW_SIZE)
                      for _ in range(n_sequences)]

        model.fit(np.concatenate(sequences))

        # Evaluate on a few sequences
        correct = 0
        for seq in sequences[:50]:
            stage_idx = int(seq[0])
            next_stage, _ = model.predict_next(stage_idx)
            if next_stage == seq[1]:
                correct += 1

        assert correct > 0

    def test_metadata_saving(self):
        """Verify model metadata is saved correctly."""
        from src.models.transformer import train_transformer

        n = 100
        X = np.random.randn(n, WINDOW_SIZE, 46).astype(np.float32)
        y = np.random.randint(0, 7, size=n).astype(np.int64)

        split = int(n * 0.8)
        metadata = train_transformer(
            X[:split], y[:split], X[split:], y[split:],
            n_features=46, d_model=32, n_heads=2, n_layers=1,
            epochs=1, batch_size=16
        )

        assert "model_type" in metadata
        assert metadata["model_type"] == "transformer"
        assert "val_accuracy" in metadata
        assert "d_ff" in metadata
        assert metadata["d_ff"] == 256


class TestBaselineComparison:
    """Tests for the baseline comparison framework."""

    def test_majority_classifier(self):
        """Verify majority classifier works correctly."""
        clf = MajorityClassifier()
        y = np.array([0] * 80 + [1] * 15 + [2] * 5)
        clf.fit(y)
        assert clf.majority_class == 0
        preds = clf.predict(np.random.randn(10, WINDOW_SIZE, 10))
        assert all(p == 0 for p in preds)

    def test_xgboost_baseline_fit(self):
        """Verify XGBoost baseline can be trained."""
        clf = XGBoostBaseline(n_stages=7, n_estimators=5)
        X = np.random.randn(30, WINDOW_SIZE * len(FEATURE_COLS))
        y = np.random.randint(0, 7, size=30)
        clf.fit(X, y)
        assert clf.model is not None
        preds = clf.predict(X)
        assert len(preds) == 30

    def test_class_weights_computation(self):
        """Verify class weights are computed correctly."""
        from src.baseline.comparison import compute_class_weights
        y = np.array([0] * 80 + [1] * 15 + [2] * 5)
        weights = compute_class_weights(y)
        assert len(weights) == 3
        assert weights[2] > weights[0]  # Minority class gets higher weight

    def test_load_split_data(self):
        """Verify split data loading works."""
        from src.baseline.comparison import load_split_data
        # This will use full dataset if split files don't exist
        X, y = load_split_data("train")
        assert X.shape[0] > 0
        assert y.shape[0] > 0
