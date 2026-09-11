"""
markov.py -- First-order Markov baseline for stage prediction.

Trains a first-order Markov chain on the sliding-window sequences
to predict the next attack stage given the current stage.
Serves as a simple baseline to compare against LSTM/Transformer models.

Usage:
    python -m src.baseline.markov
"""

import json
import os
import numpy as np
from collections import defaultdict
from typing import Dict, Tuple, Optional

try:
    from src.config import STAGE_ORDER, STAGE_TO_IDX, MODEL_DIR
except ImportError:
    STAGE_ORDER = ["Benign", "Recon", "CredAccess", "Exploit",
                   "LateralMove", "C2", "Impact"]
    STAGE_TO_IDX = {s: i for i, s in enumerate(STAGE_ORDER)}
    MODEL_DIR = "models"


class MarkovBaseline:
    """
    First-order Markov chain for attack stage prediction.

    Transition matrix T[i][j] = P(next_stage=j | current_stage=i)
    Predictions pick the argmax of the transition probabilities.
    """

    def __init__(self, n_stages: int = 7):
        self.n_stages = n_stages
        self.transition_counts: np.ndarray = np.zeros(
            (n_stages, n_stages), dtype=np.int64
        )
        self.transition_probs: np.ndarray = np.zeros(
            (n_stages, n_stages), dtype=np.float64
        )
        self.trained = False

    def fit(self, stages: np.ndarray) -> "MarkovBaseline":
        """
        Train the Markov chain on a sequence of stage labels.

        Args:
            stages: np.ndarray of integer stage labels.
                    Treated as a flat sequence: consecutive pairs
                    (stages[t], stages[t+1]) are training examples.
        """
        print(f"[markov] Training on {len(stages)} stage transitions...")

        stages = np.asarray(stages).flatten()
        for i in range(len(stages) - 1):
            curr = int(stages[i])
            next_stage = int(stages[i + 1])
            if 0 <= curr < self.n_stages and 0 <= next_stage < self.n_stages:
                self.transition_counts[curr][next_stage] += 1

        # Normalise to get probabilities
        row_sums = self.transition_counts.sum(axis=1, keepdims=True)
        # Avoid division by zero
        row_sums = np.where(row_sums == 0, 1, row_sums)
        self.transition_probs = self.transition_counts / row_sums
        self.trained = True

        print(f"[markov] Training complete. {self.n_stages}x{self.n_stages} "
              f"transition matrix learned.")
        return self

    def predict_next(self, current_stage: int) -> Tuple[int, float]:
        """
        Predict the next stage given the current stage.

        Args:
            current_stage: Integer stage label (0-6).

        Returns:
            (predicted_stage, confidence) tuple.
        """
        if not self.trained:
            raise RuntimeError("MarkovBaseline must be trained first.")
        probs = self.transition_probs[current_stage]
        pred = int(np.argmax(probs))
        conf = float(probs[pred])
        return pred, conf

    def predict_sequence(self, start_stage: int, length: int = 20) -> list:
        """
        Generate a predicted stage sequence starting from a given stage.

        Args:
            start_stage: Integer stage label to start from.
            length: Number of stages to predict.

        Returns:
            List of predicted stage integers.
        """
        if not self.trained:
            raise RuntimeError("MarkovBaseline must be trained first.")
        stages = [start_stage]
        for _ in range(length - 1):
            next_s, _ = self.predict_next(stages[-1])
            stages.append(next_s)
        return stages

    def evaluate(self, stages: np.ndarray) -> dict:
        """
        Evaluate accuracy on held-out stage sequence.

        Args:
            stages: np.ndarray of integer stage labels.

        Returns:
            Dict with accuracy and per-stage precision/recall.
        """
        stages = np.asarray(stages).flatten()
        correct = 0
        total = 0
        per_stage_correct = defaultdict(int)
        per_stage_total = defaultdict(int)

        for i in range(len(stages) - 1):
            curr = int(stages[i])
            actual = int(stages[i + 1])
            if not (0 <= curr < self.n_stages and 0 <= actual < self.n_stages):
                continue
            predicted = int(np.argmax(self.transition_probs[curr]))
            total += 1
            per_stage_total[curr] += 1
            if predicted == actual:
                correct += 1
                per_stage_correct[curr] += 1

        accuracy = correct / total if total > 0 else 0.0

        result = {
            "accuracy": round(accuracy, 4),
            "total_predictions": total,
            "correct_predictions": correct,
            "macro_f1": round(
                _macro_f1_per_stage(per_stage_correct,
                                    per_stage_total,
                                    self.n_stages), 4
            ),
            "per_stage_accuracy": {
                str(k): round(
                    v / per_stage_total[k], 4
                ) if per_stage_total[k] > 0 else 0.0
                for k, v in per_stage_correct.items()
            },
        }
        print(f"[markov] Evaluation: accuracy={accuracy:.4f}, "
              f"macro_f1={result['macro_f1']:.4f} "
              f"({correct}/{total})")
        return result


def _macro_f1_per_stage(per_stage_correct: dict,
                           per_stage_total: dict,
                           n_stages: int) -> float:
    """Compute macro-averaged F1 from per-stage counts."""
    f1s = []
    for s in range(n_stages):
        total = per_stage_total.get(s, 0)
        if total > 0:
            precision = per_stage_correct[s] / total
            recall = per_stage_correct[s] / total  # same here
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        else:
            f1 = 0.0
        f1s.append(f1)
    return sum(f1s) / len(f1s) if f1s else 0.0


def load_sequences(path: str = "data/sequences") -> Tuple[np.ndarray, np.ndarray]:
    """Load saved sequences."""
    X = np.load(os.path.join(path, "X.npy"))
    y = np.load(os.path.join(path, "y.npy"))
    return X, y


def train_and_save_markov(sequences_path: str = "data/sequences",
                          output_dir: str = MODEL_DIR) -> dict:
    """
    Train Markov baseline and save model + evaluation.

    Args:
        sequences_path: Path to saved sequences.
        output_dir: Directory to save model artifacts.

    Returns:
        Dict with training results.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load sequences
    X, y = load_sequences(sequences_path)
    print(f"[markov] Loaded {X.shape[0]} sequences from {sequences_path}")

    # Use y values as the Markov state chain
    # y[i] = stage label of the flow immediately after window i
    # So consecutive y values form a stage sequence: y[0], y[1], y[2], ...
    # Train Markov: P(y[t+1] | y[t])
    # Reshape y to (N, 1) and iterate over consecutive pairs
    model = MarkovBaseline(n_stages=len(STAGE_ORDER))
    model.fit(y)

    # Evaluate
    results = model.evaluate(y)

    # Save model
    model_path = os.path.join(output_dir, "markov_baseline.npz")
    np.savez(model_path,
             transition_counts=model.transition_counts,
             transition_probs=model.transition_probs)
    print(f"[markov] Model saved to {model_path}")

    # Save metadata
    metadata = {
        "model_type": "markov_first_order",
        "n_stages": len(STAGE_ORDER),
        "stage_to_idx": STAGE_TO_IDX,
        "idx_to_stage": {v: k for k, v in STAGE_TO_IDX.items()},
        "accuracy": results["accuracy"],
        "total_predictions": results["total_predictions"],
        "model_path": model_path,
    }
    metadata_path = os.path.join(output_dir, "markov_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # Print transition matrix
    print("\n" + "=" * 80)
    print("MARKOV TRANSITION MATRIX (trained)")
    print("=" * 80)
    header = f"{'From':<18}" + "".join(f"{s:<15}" for s in STAGE_ORDER)
    print(header)
    print("-" * 80)
    stage_names = STAGE_ORDER
    for i, stage in enumerate(stage_names):
        row = f"{stage:<18}"
        for j, _ in enumerate(stage_names):
            row += f"{model.transition_probs[i][j]:<15.4f}"
        print(row)
    print("=" * 80)

    return metadata


if __name__ == "__main__":
    metadata = train_and_save_markov()
    print(f"\nMarkov baseline trained with accuracy: {metadata['accuracy']:.4f}")
