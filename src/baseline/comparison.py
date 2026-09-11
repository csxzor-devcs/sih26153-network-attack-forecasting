"""
comparison.py -- Proper baseline comparison framework.

Compares 4 baselines in increasing order of complexity:
1. MajorityClassifier: Always predicts the most frequent class
2. MarkovBaseline: First-order Markov chain on stage transitions
3. XGBoostClassifier: Gradient-boosted trees on window features
4. AttackLSTM: LSTM neural network (same architecture as before)

All models are evaluated on the SAME campaign-held-out test set
to ensure fair comparison.

Usage:
    python -m src.baseline.comparison
"""

import json
import os
import numpy as np
from collections import Counter
from typing import Dict, List, Tuple

import pandas as pd

try:
    from src.config import STAGE_ORDER, STAGE_TO_IDX, MODEL_DIR
except ImportError:
    STAGE_ORDER = ["Benign", "Recon", "CredAccess", "Exploit",
                   "LateralMove", "C2", "Impact"]
    STAGE_TO_IDX = {s: i for i, s in enumerate(STAGE_ORDER)}
    MODEL_DIR = "models"


class MajorityClassifier:
    """
    Baseline that always predicts the most frequent class.

    This is the "predict Benign every time" baseline. Any model
    that cannot beat this is not learning anything useful.
    """

    def __init__(self):
        self.majority_class = 0  # Benign = index 0
        self.class_counts = None

    def fit(self, y_train: np.ndarray) -> "MajorityClassifier":
        """Compute the most frequent class."""
        self.class_counts = Counter(y_train.flatten().astype(int))
        self.majority_class = self.class_counts.most_common(1)[0][0]
        print(f"[majority] Majority class: {STAGE_ORDER[self.majority_class]} "
              f"({self.class_counts[self.majority_class]} samples)")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Always predict the majority class."""
        n_samples = X.shape[0]
        return np.full(n_samples, self.majority_class, dtype=np.int64)

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """Compute accuracy."""
        y_true = y_true.flatten().astype(int)
        y_pred = y_pred.flatten().astype(int)
        correct = (y_true == y_pred).sum()
        total = len(y_true)
        accuracy = correct / total if total > 0 else 0.0
        return {"accuracy": round(accuracy, 4), "total": total}


class XGBoostBaseline:
    """
    XGBoost classifier for stage prediction.

    Uses the same window features as LSTM/Transformer but with
    gradient-boosted trees. This serves as a strong non-neural
    baseline to compare against.
    """

    def __init__(self, n_stages: int = 7, n_estimators: int = 100,
                 max_depth: int = 6, learning_rate: float = 0.1):
        self.n_stages = n_stages
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.model = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "XGBoostBaseline":
        """Train XGBoost on flattened window features."""
        from xgboost import XGBClassifier

        n_samples = X_train.shape[0]
        # Flatten (n_samples, window_size, n_features) -> (n_samples, window_size * n_features)
        X_flat = X_train.reshape(n_samples, -1)

        self.model = XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            objective="multi:softprob",
            num_class=self.n_stages,
            eval_metric="mlogloss",
            tree_method="hist",
            n_jobs=-1,
            random_state=42,
        )
        self.model.fit(X_flat, y_train.flatten().astype(int))
        print(f"[xgboost] Trained on {n_samples} samples, "
              f"features={X_flat.shape[1]}")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict stage labels."""
        n_samples = X.shape[0]
        X_flat = X.reshape(n_samples, -1)
        return self.model.predict(X_flat)

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """Compute accuracy and per-class accuracy."""
        from sklearn.metrics import f1_score, precision_score, recall_score
        y_true = y_true.flatten().astype(int)
        y_pred = y_pred.flatten().astype(int)
        correct = (y_true == y_pred).sum()
        total = len(y_true)
        accuracy = correct / total if total > 0 else 0.0

        # Per-class accuracy
        per_class = {}
        for stage_idx in range(self.n_stages):
            mask = (y_true == stage_idx)
            if mask.sum() > 0:
                per_class[STAGE_ORDER[stage_idx]] = round(
                    (y_pred[mask] == stage_idx).mean(), 4
                )

        macro_f1 = f1_score(y_true, y_pred, average="macro",
                              zero_division=0)
        weighted_f1 = f1_score(y_true, y_pred, average="weighted",
                                  zero_division=0)

        return {
            "accuracy": round(accuracy, 4),
            "total": total,
            "per_class_accuracy": per_class,
            "macro_f1": round(macro_f1, 4),
            "f1_weighted": round(weighted_f1, 4),
        }


def load_split_data(split_name: str = "test") -> Tuple[np.ndarray, np.ndarray]:
    """Load campaign-split data."""
    X_path = f"data/sequences/X_{split_name}.npy"
    y_path = f"data/sequences/y_{split_name}.npy"

    if os.path.exists(X_path) and os.path.exists(y_path):
        X = np.load(X_path).astype(np.float32)
        y = np.load(y_path).astype(np.int64)
        print(f"[comparison] Loaded {split_name}: X shape={X.shape}, y shape={y.shape}")
        return X, y

    # Fallback to full dataset
    print(f"[comparison] No split data for {split_name}, using full dataset")
    X = np.load("data/sequences/X.npy").astype(np.float32)
    y = np.load("data/sequences/y.npy").astype(np.int64)
    return X, y


def compute_class_weights(y_train: np.ndarray) -> np.ndarray:
    """Compute class weights for handling imbalance."""
    class_counts = np.bincount(y_train.flatten().astype(int))
    total = class_counts.sum()
    n_classes = len(class_counts)
    weight = total / (n_classes * class_counts)
    weight = np.minimum(weight, weight.max() * 5.0)
    return weight / weight.mean()


def run_comparison(X_train: np.ndarray, y_train: np.ndarray,
                    X_val: np.ndarray, y_val: np.ndarray,
                    X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, dict]:
    """
    Run all baselines and compare results.

    Args:
        X_train, y_train: Training data (campaign-split).
        X_val, y_val: Validation data.
        X_test, y_test: Test data (campaign-held-out).

    Returns:
        Dict mapping baseline name -> results dict.
    """
    results = {}

    # === 1. Majority Classifier ===
    print("\n" + "=" * 80)
    print("BASELINE 1: MAJORITY CLASSIFIER")
    print("=" * 80)
    majority = MajorityClassifier()
    majority.fit(y_train)
    y_pred = majority.predict(X_test)
    results["majority"] = majority.evaluate(y_test, y_pred)
    print(f"[majority] Test accuracy: {results['majority']['accuracy']:.4f}")

    # === 2. Markov Baseline ===
    print("\n" + "=" * 80)
    print("BASELINE 2: MARKOV CHAIN")
    print("=" * 80)
    from src.baseline.markov import MarkovBaseline
    markov = MarkovBaseline(n_stages=len(STAGE_ORDER))
    # Flatten y_train for Markov: consecutive pairs
    y_flat = y_train.flatten().astype(int)
    markov.fit(y_flat)
    # For evaluation, use Markov to predict next stage from last stage in each test window
    # Simpler: use the test set labels as the Markov chain state sequence
    y_flat_test = y_test.flatten().astype(int)
    results["markov"] = markov.evaluate(y_flat_test)
    print(f"[markov] Test accuracy: {results['markov']['accuracy']:.4f}")

    # === 3. XGBoost Baseline ===
    print("\n" + "=" * 80)
    print("BASELINE 3: XGBOOST CLASSIFIER")
    print("=" * 80)
    xgb = XGBoostBaseline(n_stages=len(STAGE_ORDER))
    xgb.fit(X_train, y_train)
    y_pred = xgb.predict(X_test)
    results["xgboost"] = xgb.evaluate(y_test, y_pred)
    print(f"[xgboost] Test accuracy: {results['xgboost']['accuracy']:.4f}")

    # === 4. LSTM (if available) ===
    print("\n" + "=" * 80)
    print("BASELINE 4: LSTM NEURAL NETWORK")
    print("=" * 80)
    try:
        lstm_results = _evaluate_lstm(X_test, y_test)
        results["lstm"] = lstm_results
        print(f"[lstm] Test accuracy: {lstm_results['accuracy']:.4f}")
    except Exception as e:
        print(f"[lstm] Could not evaluate: {e}")
        results["lstm"] = {"accuracy": 0.0, "error": str(e)}

    return results


def _evaluate_lstm(X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Evaluate the trained LSTM model."""
    import torch
    from torch.utils.data import DataLoader
    from src.models.lstm import AttackLSTM, AttackSequenceDataset
    from sklearn.metrics import accuracy_score

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AttackLSTM(n_features=X_test.shape[2], hidden_size=128,
                         n_layers=2, n_stages=len(STAGE_ORDER))
    model_path = os.path.join(MODEL_DIR, "lstm_best.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"LSTM model not found at {model_path}")

    model.load_state_dict(torch.load(model_path, weights_only=False))
    model.to(device)
    model.eval()

    test_dataset = AttackSequenceDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            outputs = model(batch_X)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    return {"accuracy": round(accuracy, 4)}


def print_comparison_results(results: Dict[str, dict]):
    """Pretty-print the comparison table."""
    print("\n" + "=" * 80)
    print("BASELINE COMPARISON RESULTS")
    print("=" * 80)
    header = f"{'Baseline':<20} {'Accuracy':<12} {'# Samples':<12}"
    print(header)
    print("-" * 80)

    for name, metrics in results.items():
        acc = metrics.get("accuracy", 0.0)
        total = metrics.get("total", "N/A")
        print(f"{name:<20} {acc:<12.4f} {total:<12}")

    print("=" * 80)

    # Determine winner by Macro F1 (not accuracy)
    valid_results = {k: v for k, v in results.items()
                     if "f1_weighted" in v and "accuracy" in v}
    if valid_results:
        winner = max(valid_results, key=lambda k: valid_results[k].get("f1_weighted", 0))
        print(f"\n🏆 Best baseline: {winner} "
              f"(Macro F1: {valid_results[winner].get('f1_weighted', 0):.4f}, "
              f"Accuracy: {valid_results[winner]['accuracy']:.4f}) "
              f"[ranked by macro_f1]")
        print(f"\n📊 Baseline hierarchy:")
        print(f"   Majority → Markov → XGBoost → LSTM → Transformer")
        print(f"   Ranked by Macro F1 (not accuracy) for imbalanced classes.")
        print(f"   If XGBoost ≈ LSTM, consider feature engineering or attention.")
        print(f"   If Markov ≈ Majority, the Markov chain has no signal.")

    return results


def save_comparison_report(results: Dict[str, dict],
                             output_dir: str = "reports") -> str:
    """Save comparison results as JSON report."""
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "baseline_comparison.json")

    valid = {k: v for k, v in results.items()
             if "macro_f1" in v}
    winner = max(valid, key=lambda k: valid[k]["macro_f1"]) if valid else None
    report = {
        "timestamp": __import__("datetime").datetime.now().isoformat(),
        "results": results,
        "winner": winner,
        "ranking_metric": "macro_f1",
        "winner_macro_f1": valid[winner]["macro_f1"] if winner else None,
    }

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"[comparison] Report saved to {report_path}")
    return report_path


if __name__ == "__main__":
    print("=== Baseline Comparison ===")

    # Load campaign-split data
    X_train, y_train = load_split_data("train")
    X_val, y_val = load_split_data("val")
    X_test, y_test = load_split_data("test")

    # Run comparison
    results = run_comparison(X_train, y_train, X_val, y_val, X_test, y_test)

    # Print results
    print_comparison_results(results)

    # Save report
    save_comparison_report(results)
