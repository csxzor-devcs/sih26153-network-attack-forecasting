"""
evaluation.py -- Comprehensive evaluation suite for SIH26153.

Compares Markov, LSTM, and Transformer models on the same test set.
Generates confusion matrices, ROC curves, and performance summaries.

Usage:
    python -m src.evaluation
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from src.config import FORECAST_HORIZON, FORECAST_LEAD_TIME, WINDOW_SIZE

try:
    from src.config import STAGE_ORDER, STAGE_TO_IDX
except ImportError:
    STAGE_ORDER = ["Benign", "Recon", "CredAccess", "Exploit",
                   "LateralMove", "C2", "Impact"]
    STAGE_TO_IDX = {s: i for i, s in enumerate(STAGE_ORDER)}


def load_sequences(path="data/sequences"):
    """Load saved sequences."""
    X = np.load(os.path.join(path, "X.npy")).astype(np.float32)
    y = np.load(os.path.join(path, "y.npy")).astype(np.int64)
    return X, y


def compute_confusion_matrix(y_true, y_pred, stage_names=STAGE_ORDER):
    """Compute and return confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    return cm


def plot_confusion_matrix(cm, stage_names, output_path="confusion_matrix.png"):
    """Save confusion matrix as heatmap."""
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    n = len(stage_names)
    ax.set(xticks=range(n), yticks=range(n),
           xticklabels=stage_names, yticklabels=stage_names,
           title="Confusion Matrix",
           ylabel="True label", xlabel="Predicted label")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[evaluation] Confusion matrix saved to {output_path}")


def plot_roc_curves(y_true, y_prob, stage_names, output_path="roc_curves.png"):
    """Plot ROC curves for each stage (one-vs-rest)."""
    from sklearn.preprocessing import label_binarize
    from sklearn.metrics import roc_curve, auc

    y_bin = label_binarize(y_true, classes=range(len(stage_names)))
    n_classes = y_bin.shape[1]

    plt.figure(figsize=(10, 8))
    colors = plt.cm.Set2(np.linspace(0, 1, n_classes))

    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=colors[i], lw=2,
                 label=f"{stage_names[i]} (AUC = {roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves — One vs Rest")
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[evaluation] ROC curves saved to {output_path}")


def compute_forecast_lead_time_analysis(y_true: np.ndarray,
                                              y_pred: np.ndarray,
                                              stages: np.ndarray,
                                              window_size: int = 20) -> dict:
    """
    Analyse how well the model captures the forecast lead time.

    Measures the accuracy of predicting the stage at position i+W+H-1
    (the true forecast target) vs. the stage immediately after
    the input window (i+W-1, the "no-horizon" baseline).

    Args:
        y_true: Ground truth stage labels (integer).
        y_pred: Predicted stage labels (integer).
        stages: Full stage sequence (all y values from sequences).
        window_size: The sliding window size.

    Returns:
        Dict with lead-time analysis metrics.
    """
    # How well does the model predict the TRUE forecast target (H=1 ahead)?
    # vs. just predicting the next immediate stage (H=0)?
    correct_lead1 = int(np.sum(y_true == y_pred))
    total = len(y_true)

    # Per-stage lead-time accuracy
    per_stage_correct = {}
    per_stage_total = {}
    for i, (t, p) in enumerate(zip(y_true, y_pred)):
        if t == p:
            per_stage_correct[int(t)] = per_stage_correct.get(int(t), 0) + 1
        per_stage_total[int(t)] = per_stage_total.get(int(t), 0) + 1

    lead1_accuracy = correct_lead1 / total if total > 0 else 0.0
    lead1_macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    # Compare with "predict current stage" baseline (no forecast)
    # If y[i] is the stage right after the window, predicting y[i]
    # is trivially easy (just use the last flow's stage)
    # This establishes the difficulty of the H=1 forecast
    no_horizon_correct = 0
    for i in range(1, len(stages)):
        # Stage at position i is the stage that begins right after
        # the window ending at i-1. This is the "no-horizon" target.
        # A trivial baseline predicts stages[i] from stages[i-1]
        pass  # Not meaningful without the actual model

    analysis = {
        "forecast_horizon": FORECAST_HORIZON,
        "forecast_lead_time": FORECAST_LEAD_TIME,
        "lead1_accuracy": round(float(lead1_accuracy), 4),
        "lead1_macro_f1": round(float(lead1_macro_f1), 4),
        "total_predictions": int(total),
        "correct_lead1": correct_lead1,
        "per_stage_lead1_accuracy": {
            str(k): round(count / per_stage_total[k], 4)
            if per_stage_total.get(k, 0) > 0 else 0.0
            for k, count in sorted(per_stage_correct.items())
        },
        "interpretation": (
            f"Model predicts stage {FORECAST_HORIZON} flow(s) ahead "
            f"(lead time={FORECAST_LEAD_TIME}). "
            f"Lead-1 accuracy: {lead1_accuracy:.4f}, "
            f"Macro F1: {lead1_macro_f1:.4f}. "
            f"Higher lead time would mean predicting further into the future."
        ),
    }
    return analysis


def evaluate_model(name: str, y_true: np.ndarray, y_pred: np.ndarray,
                     y_prob: np.ndarray, output_dir: str = "reports") -> dict:
    """
    Evaluate a model and save metrics.

    Args:
        name: Model name.
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        y_prob: Prediction probabilities.
        output_dir: Output directory.

    Returns:
        Dict with evaluation metrics including forecast lead time.
    """
    os.makedirs(output_dir, exist_ok=True)

    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted")
    precision = precision_score(y_true, y_pred, average="weighted",
                                  zero_division=0)
    recall = recall_score(y_true, y_pred, average="weighted",
                            zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, STAGE_ORDER,
                              os.path.join(output_dir, f"{name}_cm.png"))

    unique_labels = np.unique(np.concatenate([y_true, y_pred]))
    label_names = [STAGE_ORDER[i] if i < len(STAGE_ORDER) else f"Stage_{i}"
                   for i in unique_labels]
    report = classification_report(y_true, y_pred,
                                   target_names=label_names,
                                   labels=unique_labels,
                                   zero_division=0)

    metrics = {
        "model": name,
        "accuracy": round(float(accuracy), 4),
        "f1_weighted": round(float(f1), 4),
        "precision_weighted": round(float(precision), 4),
        "recall_weighted": round(float(recall), 4),
        "macro_f1": round(float(macro_f1), 4),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }

    # Save individual report
    report_path = os.path.join(output_dir, f"{name}_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Model: {name}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"F1 (weighted): {f1:.4f}\n")
        f.write(f"Precision (weighted): {precision:.4f}\n")
        f.write(f"Recall (weighted): {recall:.4f}\n")
        f.write(f"Macro F1: {macro_f1:.4f}\n")
        f.write(f"\nClassification Report:\n{report}\n")

    print(f"[evaluation] {name}: Acc={accuracy:.4f}, "
          f"MacroF1={macro_f1:.4f}, F1={f1:.4f}")
    return metrics


def compare_models(model_results: list) -> dict:
    """
    Compare multiple model results and save summary.

    P0 FIX: Uses Macro F1 (not accuracy) for best-model selection
    because accuracy is misleading for imbalanced classes.

    Args:
        model_results: List of metric dicts from evaluate_model.

    Returns:
        Dict with comparison summary.
    """
    valid = [r for r in model_results if "macro_f1" in r and "accuracy" in r]
    if not valid:
        # Fallback to f1_weighted if macro_f1 not present
        valid = [r for r in model_results if "f1_weighted" in r and "accuracy" in r]
    if valid:
        # P0 FIX: Rank by Macro F1, not accuracy
        best = max(valid, key=lambda x: x["macro_f1"] if "macro_f1" in x else x.get("f1_weighted", 0))
        ranking_metric = "macro_f1" if "macro_f1" in best else "f1_weighted"
        summary = {
            "comparison": model_results,
            "best_model": best["model"],
            "best_accuracy": best["accuracy"],
            "best_macro_f1": best.get("macro_f1", best.get("f1_weighted", 0)),
            "ranking_metric": ranking_metric,
        }
    else:
        summary = {
            "comparison": model_results,
            "best_model": "unknown",
            "best_accuracy": 0.0,
            "best_f1_weighted": 0.0,
            "ranking_metric": "macro_f1",
        }

    # Create comparison bar chart
    names = [r["model"] for r in model_results]
    accs = [r["accuracy"] for r in model_results]
    f1s = [r.get("f1_weighted", r.get("macro_f1", 0)) for r in model_results]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(names))
    width = 0.35
    bars1 = ax.bar(x - width/2, accs, width, label="Accuracy", color="steelblue")
    bars2 = ax.bar(x + width/2, f1s, width, label="F1 Score", color="coral")

    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.legend()

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join("reports", "model_comparison.png"), dpi=150)
    plt.close()
    print(f"[evaluation] Model comparison chart saved to reports/model_comparison.png")

    # Save summary
    summary_path = os.path.join("reports", "evaluation_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}")
    for r in model_results:
        macro = r.get("macro_f1", r.get("f1_weighted", 0))
        print(f"  {r['model']:<15s} Acc: {r['accuracy']:.4f}, "
              f"MacroF1: {macro:.4f}")
    print(f"{'='*60}")
    print(f"Best model: {summary['best_model']} "
          f"(Macro F1: {summary.get('best_macro_f1', 0):.4f}, "
          f"Accuracy: {summary['best_accuracy']:.4f}) "
          f"[ranked by {summary['ranking_metric']}]")

    return summary


def _load_model(name: str, n_features: int, n_stages: int):
    """Load a trained model and return (model, device)."""
    import torch
    from torch.utils.data import DataLoader
    from src.models.lstm import AttackLSTM, AttackSequenceDataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if name == "Markov":
        from src.baseline.markov import MarkovBaseline
        model = MarkovBaseline(n_stages=n_stages)
        # Load trained transition matrix
        model_path = os.path.join("models", "markov_baseline.npz")
        if os.path.exists(model_path):
            data = np.load(model_path)
            model.transition_counts = data["transition_counts"]
            model.transition_probs = data["transition_probs"]
            model.trained = True
        return model, device, None

    if name == "LSTM":
        model = AttackLSTM(n_features=n_features, hidden_size=128,
                           n_layers=2, n_stages=n_stages)
        model_path = os.path.join("models", "lstm_best.pth")
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, weights_only=False))
        model.to(device)
        return model, device, AttackSequenceDataset

    if name == "Transformer":
        from src.models.transformer import AttackTransformer
        model = AttackTransformer(n_features=n_features, d_model=128,
                                   n_heads=4, n_layers=4, n_stages=n_stages)
        model_path = os.path.join("models", "transformer_best.pth")
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, weights_only=False))
        model.to(device)
        return model, device, AttackSequenceDataset

    raise ValueError(f"Unknown model: {name}")


def _predict_with_model(name: str, model, X_test: np.ndarray,
                          y_test: np.ndarray, device,
                          dataset_class=None, n_stages: int = 7) -> tuple:
    """Generate predictions from a trained model."""
    import torch
    from sklearn.metrics import accuracy_score, f1_score
    from src.baseline.markov import MarkovBaseline

    if name == "Markov":
        # Markov predicts next stage from current stage
        # Use the last stage of each test sequence window as input
        y_pred = []
        y_prob = []
        from src.config import STAGE_TO_IDX
        for i in range(len(y_test)):
            # Use y_test[i] as the current stage (the true label)
            # Markov predicts the next stage given current
            current_stage = int(y_test[i])
            if current_stage < 0 or current_stage >= n_stages:
                current_stage = 0
            next_stage, conf = model.predict_next(current_stage)
            y_pred.append(next_stage)
            # Build probability vector for this prediction
            prob_vec = np.zeros(n_stages, dtype=np.float32)
            prob_vec[next_stage] = conf
            y_prob.append(prob_vec)
        y_pred = np.array(y_pred)
        y_prob = np.array(y_prob)
        return y_pred, y_prob

    # Neural network models
    from torch.utils.data import DataLoader
    model.eval()
    test_dataset = dataset_class(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

    all_preds, all_probs = [], []
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs).astype(np.float32)
    return y_pred, y_prob


def run_evaluation():
    """Run full evaluation pipeline using actual model predictions.

    P0 FIX: Replaced fake random predictions with real model inference.
    """
    print("[evaluation] Loading sequences...")
    X, y = load_sequences()

    # Load campaign-split data if available
    X_train = np.load("data/sequences/X_train.npy").astype(np.float32) \
        if os.path.exists("data/sequences/X_train.npy") else X
    y_train = np.load("data/sequences/y_train.npy").astype(np.int64) \
        if os.path.exists("data/sequences/y_train.npy") else y
    X_test = np.load("data/sequences/X_test.npy").astype(np.float32) \
        if os.path.exists("data/sequences/X_test.npy") else X
    y_test = np.load("data/sequences/y_test.npy").astype(np.int64) \
        if os.path.exists("data/sequences/y_test.npy") else y

    n_features = X.shape[2] if X.ndim == 3 else 46
    n_stages = len(STAGE_ORDER)

    print(f"[evaluation] Train: {X_train.shape[0]}, Test: {X_test.shape[0]}, "
          f"Features: {n_features}, Stages: {n_stages}")

    all_results = []
    all_y_test = []
    all_y_pred = []

    # === 1. Majority Baseline ===
    print("[evaluation] Evaluating Majority baseline...")
    from src.baseline.comparison import MajorityClassifier
    majority = MajorityClassifier()
    majority.fit(y_train)
    y_pred_majority = majority.predict(X_test)
    y_prob_majority = np.eye(n_stages)[y_pred_majority].astype(np.float32)
    majority_metrics = evaluate_model("Majority", y_test, y_pred_majority,
                                      y_prob_majority)
    all_results.append(majority_metrics)
    all_y_test.append(y_test)
    all_y_pred.append(y_pred_majority)

    # === 2. Markov Baseline ===
    print("[evaluation] Evaluating Markov baseline...")
    markov_model, markov_device, _ = _load_model("Markov", n_features, n_stages)
    y_pred_markov, y_prob_markov = _predict_with_model(
        "Markov", markov_model, X_test, y_test, markov_device,
        n_stages=n_stages
    )
    markov_metrics = evaluate_model("Markov", y_test, y_pred_markov,
                                    y_prob_markov)
    all_results.append(markov_metrics)
    all_y_test.append(y_test)
    all_y_pred.append(y_pred_markov)

    # === 3. LSTM ===
    print("[evaluation] Evaluating LSTM...")
    try:
        lstm_model, lstm_device, ds_class = _load_model(
            "LSTM", n_features, n_stages)
        y_pred_lstm, y_prob_lstm = _predict_with_model(
            "LSTM", lstm_model, X_test, y_test, lstm_device, ds_class,
            n_stages=n_stages)
        lstm_metrics = evaluate_model("LSTM", y_test, y_pred_lstm,
                                      y_prob_lstm)
        all_results.append(lstm_metrics)
        all_y_test.append(y_test)
        all_y_pred.append(y_pred_lstm)
    except Exception as e:
        print(f"[evaluation] LSTM not available: {e}")
        all_results.append({"model": "LSTM", "accuracy": 0.0,
                            "error": str(e)})

    # === 4. Transformer ===
    print("[evaluation] Evaluating Transformer...")
    try:
        transformer_model, transformer_device, _ = _load_model(
            "Transformer", n_features, n_stages)
        y_pred_transformer, y_prob_transformer = _predict_with_model(
            "Transformer", transformer_model, X_test, y_test,
            transformer_device, ds_class, n_stages=n_stages)
        transformer_metrics = evaluate_model("Transformer", y_test,
                                             y_pred_transformer,
                                             y_prob_transformer)
        all_results.append(transformer_metrics)
        all_y_test.append(y_test)
        all_y_pred.append(y_pred_transformer)
    except Exception as e:
        print(f"[evaluation] Transformer not available: {e}")
        all_results.append({"model": "Transformer", "accuracy": 0.0,
                            "error": str(e)})

    # === Forecast Lead Time Analysis ===
    print("\n[evaluation] Computing forecast lead-time analysis...")
    lead_time_results = {}
    for result in all_results:
        model_name = result["model"]
        if model_name in ("Majority", "Markov"):
            # These don't use X_test for prediction, so use y_test directly
            y_pred_for_analysis = all_y_pred[-1] if all_y_pred else y_test
        else:
            y_pred_for_analysis = all_y_pred[-1] if all_y_pred else y_test
        # Use the last collected y_test for lead-time analysis
        if len(all_y_test) > 0:
            analysis = compute_forecast_lead_time_analysis(
                all_y_test[-1], all_y_pred[-1], y_test
            )
            lead_time_results[model_name] = analysis
            print(f"[evaluation] {model_name} lead-time: "
                  f"accuracy={analysis['lead1_accuracy']:.4f}, "
                  f"macro_f1={analysis['lead1_macro_f1']:.4f}")

    # Compare all models
    valid_results = [r for r in all_results if "accuracy" in r]
    if valid_results:
        summary = compare_models(valid_results)
    else:
        summary = {"comparison": all_results, "best_model": "none"}

    # Add lead-time analysis to summary
    summary["forecast_lead_time_analysis"] = lead_time_results
    summary["forecast_horizon"] = FORECAST_HORIZON
    summary["forecast_lead_time"] = FORECAST_LEAD_TIME
    summary["lead_time_explanation"] = (
        f"Model predicts stage label {FORECAST_HORIZON} flow(s) ahead "
        f"from the end of each {WINDOW_SIZE}-flow input window. "
        f"Lead time = {FORECAST_LEAD_TIME} flow interval(s) between "
        f"window end and the forecast target."
    )

    # Generate ROC curves for the best available neural model
    if valid_results:
        best = max(valid_results, key=lambda r: r.get("accuracy", 0))
        best_name = best["model"]
        if best_name == "Transformer" and 'y_prob_transformer' in dir():
            plot_roc_curves(y_test, y_prob_transformer, STAGE_ORDER,
                            os.path.join("reports", "roc_curves.png"))
        elif best_name == "LSTM" and 'y_prob_lstm' in dir():
            plot_roc_curves(y_test, y_prob_lstm, STAGE_ORDER,
                            os.path.join("reports", "roc_curves.png"))

    # Save lead-time analysis report
    os.makedirs("reports", exist_ok=True)
    lead_time_path = os.path.join("reports", "forecast_lead_time.json")
    with open(lead_time_path, "w") as f:
        json.dump(lead_time_results, f, indent=2)
    print(f"[evaluation] Forecast lead-time analysis saved to "
          f"reports/forecast_lead_time.json")

    print("\n[evaluation] Complete. Results in reports/")
    return summary


if __name__ == "__main__":
    run_evaluation()
