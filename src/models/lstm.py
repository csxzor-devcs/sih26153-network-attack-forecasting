"""
lstm.py -- LSTM model for attack stage forecasting.

Trains a LSTM network on sliding-window sequences to predict
the next attack stage given a window of previous flows.
Compares against the Markov baseline from Phase 2.

Memory-efficient: uses float32 throughout, batch training.

Usage:
    python -m src.models.lstm
"""

import json
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, accuracy_score

try:
    from src.config import STAGE_ORDER, STAGE_TO_IDX, FEATURE_COLS, MODEL_DIR
except ImportError:
    STAGE_ORDER = ["Benign", "Recon", "CredAccess", "Exploit",
                   "LateralMove", "C2", "Impact"]
    STAGE_TO_IDX = {s: i for i, s in enumerate(STAGE_ORDER)}
    FEATURE_COLS = []
    MODEL_DIR = "models"


class AttackSequenceDataset(Dataset):
    """PyTorch dataset for attack sequence forecasting."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class AttackLSTM(nn.Module):
    """
    LSTM model for multi-stage attack forecasting.

    Input: (batch, window_size, n_features)
    Output: (batch, n_stages) — softmax over stage labels.
    """

    def __init__(self, n_features: int = 46,
                 hidden_size: int = 128,
                 n_layers: int = 2,
                 n_stages: int = 7,
                 dropout: float = 0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, n_stages)

    def forward(self, x):
        # x shape: (batch, seq_len, n_features)
        lstm_out, (h_n, c_n) = self.lstm(x)
        # Use last hidden state
        last_out = lstm_out[:, -1, :]  # (batch, hidden_size)
        out = self.dropout(last_out)
        out = self.fc(out)  # (batch, n_stages)
        return out


def train_lstm(X_train: np.ndarray, y_train: np.ndarray,
               X_val: np.ndarray, y_val: np.ndarray,
               n_features: int = 46,
               hidden_size: int = 128,
               n_layers: int = 2,
               epochs: int = 20,
               batch_size: int = 512,
               learning_rate: float = 0.001,
               model_dir: str = MODEL_DIR) -> dict:
    """
    Train an LSTM model on attack sequences.

    Args:
        X_train: Training features (N, window_size, n_features).
        y_train: Training labels (N,) — integer stage labels.
        X_val: Validation features.
        y_val: Validation labels.
        n_features: Number of input features.
        hidden_size: LSTM hidden dimension.
        n_layers: Number of LSTM layers.
        epochs: Number of training epochs.
        batch_size: Batch size.
        learning_rate: Optimizer learning rate.
        model_dir: Directory to save model.

    Returns:
        Dict with training results including val_accuracy.
    """
    os.makedirs(model_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[lstm] Training on {device}")

    # Create datasets and loaders
    train_dataset = AttackSequenceDataset(X_train, y_train)
    val_dataset = AttackSequenceDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=0)

    # Initialize model
    model = AttackLSTM(n_features=n_features, hidden_size=hidden_size,
                       n_layers=n_layers, n_stages=len(STAGE_ORDER))
    model.to(device)

    # Standard CrossEntropyLoss (class imbalance handled by the data skew itself)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5
    )

    best_val_acc = 0.0
    history = {"train_loss": [], "val_loss": [], "val_accuracy": []}

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[lstm] Model: {n_params} params")
    print(f"[lstm] Training for {epochs} epochs, "
          f"batch_size={batch_size}")
    print(f"[lstm] Progress: epoch | train_loss | val_loss | val_acc | elapsed")
    print(f"[lstm] {'-' * 60}")

    import time
    start_time = time.time()

    for epoch in range(epochs):
        epoch_start = time.time()

        # Training
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_X.size(0)

        train_loss /= len(train_dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item() * batch_X.size(0)
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())

        val_loss /= len(val_dataset)
        val_acc = accuracy_score(all_labels, all_preds)
        scheduler.step(val_loss)

        history["train_loss"].append(round(train_loss, 4))
        history["val_loss"].append(round(val_loss, 4))
        history["val_accuracy"].append(round(val_acc, 4))

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(model_dir, "lstm_best.pth"))
            print(f"  *** new best model saved (val_acc={val_acc:.4f})")

        elapsed = time.time() - epoch_start
        print(f"[lstm] Epoch {epoch+1:3d}/{epochs} | "
              f"{train_loss:.4f} | {val_loss:.4f} | "
              f"{val_acc:.4f} | {elapsed:.1f}s")

    # Load best model
    model.load_state_dict(
        torch.load(os.path.join(model_dir, "lstm_best.pth"),
                   weights_only=False)
    )

    # Final evaluation
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())

    final_acc = accuracy_score(all_labels, all_preds)
    print(f"[lstm] Final Val Accuracy: {final_acc:.4f}")

    # Save metadata
    metadata = {
        "model_type": "lstm",
        "n_features": n_features,
        "hidden_size": hidden_size,
        "n_layers": n_layers,
        "n_stages": len(STAGE_ORDER),
        "stage_to_idx": STAGE_TO_IDX,
        "idx_to_stage": {v: k for k, v in STAGE_TO_IDX.items()},
        "val_accuracy": round(final_acc, 4),
        "best_val_accuracy": round(best_val_acc, 4),
        "epochs": epochs,
        "history": history,
        "model_path": os.path.join(model_dir, "lstm_best.pth"),
    }
    metadata_path = os.path.join(model_dir, "lstm_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # Print classification report
    # Get actual labels present in predictions
    all_labels_arr = np.array(all_labels)
    all_preds_arr = np.array(all_preds)
    all_labels_flat = all_labels_arr.flatten()
    all_preds_flat = all_preds_arr.flatten()
    unique_labels = np.unique(np.concatenate([all_labels_flat, all_preds_flat]))
    label_names = [STAGE_ORDER[i] if i < len(STAGE_ORDER) else f"Stage_{i}"
                   for i in unique_labels]
    print("\n" + "=" * 80)
    print("LSTM CLASSIFICATION REPORT (validation)")
    print("=" * 80)
    print(classification_report(all_labels_flat, all_preds_flat,
                                 target_names=label_names,
                                 zero_division=0,
                                 labels=unique_labels))
    print("=" * 80)

    return metadata


if __name__ == "__main__":
    # Load sequences
    X = np.load("data/sequences/X.npy").astype(np.float32)
    y = np.load("data/sequences/y.npy").astype(np.int64)

    print(f"Loaded {X.shape[0]} sequences, shape: {X.shape}")

    # Subsample to 30K for fast training
    n = len(y)
    sample_size = min(30000, n)
    indices = np.random.RandomState(42).choice(n, sample_size, replace=False)
    X = X[indices]
    y = y[indices]

    # Split 80/20 train/val
    split = int(len(y) * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    print(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}")

    metadata = train_lstm(X_train, y_train, X_val, y_val,
                          n_features=X.shape[2],
                          hidden_size=128,
                          n_layers=2,
                          epochs=20,
                          batch_size=512)
    print(f"\nLSTM training complete. Val accuracy: {metadata['val_accuracy']:.4f}")
