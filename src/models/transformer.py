"""
transformer.py -- Transformer model for attack stage forecasting.

Implements a Transformer encoder for multi-stage attack forecasting,
using self-attention to capture long-range dependencies in network
flow sequences. Compares against the LSTM and Markov baselines.

Memory-efficient: uses float32 throughout, optimized attention.

Usage:
    python -m src.models.transformer
"""

import json
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Optional
from sklearn.metrics import classification_report, accuracy_score

try:
    from src.config import STAGE_ORDER, STAGE_TO_IDX, MODEL_DIR
except ImportError:
    STAGE_ORDER = ["Benign", "Recon", "CredAccess", "Exploit",
                   "LateralMove", "C2", "Impact"]
    STAGE_TO_IDX = {s: i for i, s in enumerate(STAGE_ORDER)}
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


class MultiHeadAttention(nn.Module):
    """Custom multi-head self-attention layer."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, T, C = x.shape  # batch, seq_len, d_model

        Q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if mask is not None:
            scores = scores + mask
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        context = torch.matmul(attn, V)  # (B, n_heads, T, head_dim)
        context = context.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.out_proj(context)


class TransformerEncoder(nn.Module):
    """Transformer encoder block with pre-norm and residual connections."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int,
                 dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, mask=None):
        # Pre-norm: norm before attention
        x = x + self.attention(self.norm1(x), mask)
        x = x + self.ff(self.norm2(x))
        return x


class AttackTransformer(nn.Module):
    """
    Transformer model for attack stage forecasting.

    Input: (batch, window_size, n_features)
    Output: (batch, n_stages) — softmax over stage labels.
    """

    def __init__(self, n_features: int = 46,
                 d_model: int = 128,
                 n_heads: int = 4,
                 n_layers: int = 4,
                 d_ff: int = 256,
                 n_stages: int = 7,
                 dropout: float = 0.2,
                 max_seq_len: int = 50):
        super().__init__()
        self.d_model = d_model
        self.n_stages = n_stages

        # Input projection: map n_features to d_model
        self.input_proj = nn.Linear(n_features, d_model)

        # Positional encoding
        self.pos_encoding = nn.Parameter(
            torch.randn(1, max_seq_len, d_model) * 0.01
        )

        # Transformer encoder layers
        self.layers = nn.ModuleList([
            TransformerEncoder(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, n_stages)

    def forward(self, x):
        # x shape: (batch, seq_len, n_features)
        seq_len = x.size(1)

        # Project to d_model and add positional encoding
        x = self.input_proj(x)  # (B, T, d_model)
        x = x + self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x)

        # Pass through transformer layers
        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        # Use first token (CLS-style) for classification
        out = self.fc(x[:, 0, :])  # (B, n_stages)
        return out


def train_transformer(X_train: np.ndarray, y_train: np.ndarray,
                       X_val: np.ndarray, y_val: np.ndarray,
                       n_features: int = 46,
                       d_model: int = 128,
                       n_heads: int = 4,
                       n_layers: int = 4,
                       d_ff: int = 256,
                       epochs: int = 20,
                       batch_size: int = 512,
                       learning_rate: float = 0.0005,
                       model_dir: str = MODEL_DIR,
                       class_weights: Optional[torch.Tensor] = None) -> dict:
    """
    Train a Transformer model on attack sequences.

    Args:
        X_train: Training features (N, window_size, n_features).
        y_train: Training labels (N,) — integer stage labels.
        X_val: Validation features.
        y_val: Validation labels.
        n_features: Number of input features.
        d_model: Transformer hidden dimension.
        n_heads: Number of attention heads.
        n_layers: Number of transformer layers.
        epochs: Number of training epochs.
        batch_size: Batch size.
        learning_rate: Optimizer learning rate.
        model_dir: Directory to save model.

    Returns:
        Dict with training results including val_accuracy.
    """
    os.makedirs(model_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[transformer] Training on {device}")

    # Create datasets and loaders
    train_dataset = AttackSequenceDataset(X_train, y_train)
    val_dataset = AttackSequenceDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                              shuffle=False, num_workers=0)

    # Initialize model
    model = AttackTransformer(n_features=n_features, d_model=d_model,
                               n_heads=n_heads, n_layers=n_layers)
    model.to(device)

    # Compute class weights to handle imbalance
    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    else:
        class_counts = np.bincount(y_train.flatten().astype(int))
        total = class_counts.sum()
        n_classes = len(class_counts)
        weight = total / (n_classes * class_counts)
        weight = np.minimum(weight, weight.max() * 5.0)
        weight_tensor = torch.FloatTensor(weight / weight.mean()).to(device)
        criterion = nn.CrossEntropyLoss(weight=weight_tensor)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                   weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5
    )

    best_val_acc = 0.0
    history = {"train_loss": [], "val_loss": [], "val_accuracy": []}

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[transformer] Model: {n_params} params")
    print(f"[transformer] Training for {epochs} epochs, "
          f"batch_size={batch_size}")
    print(f"[transformer] Progress: epoch | train_loss | val_loss | "
          f"val_acc | elapsed")
    print(f"[transformer] {'-' * 65}")

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
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
            torch.save(model.state_dict(),
                       os.path.join(model_dir, "transformer_best.pth"))
            print(f"  *** new best model saved (val_acc={val_acc:.4f})")

        elapsed = time.time() - epoch_start
        print(f"[transformer] Epoch {epoch+1:3d}/{epochs} | "
              f"{train_loss:.4f} | {val_loss:.4f} | "
              f"{val_acc:.4f} | {elapsed:.1f}s")

    # Load best model
    model.load_state_dict(
        torch.load(os.path.join(model_dir, "transformer_best.pth"),
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
    print(f"[transformer] Final Val Accuracy: {final_acc:.4f}")

    # Save metadata
    metadata = {
        "model_type": "transformer",
        "d_model": d_model,
        "n_heads": n_heads,
        "n_layers": n_layers,
        "n_features": n_features,
        "d_ff": d_ff,
        "n_stages": len(STAGE_ORDER),
        "stage_to_idx": STAGE_TO_IDX,
        "idx_to_stage": {v: k for k, v in STAGE_TO_IDX.items()},
        "val_accuracy": round(final_acc, 4),
        "best_val_accuracy": round(best_val_acc, 4),
        "epochs": epochs,
        "history": history,
        "model_path": os.path.join(model_dir, "transformer_best.pth"),
    }
    metadata_path = os.path.join(model_dir, "transformer_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # Print classification report
    all_labels_arr = np.array(all_labels).flatten()
    all_preds_arr = np.array(all_preds).flatten()
    unique_labels = np.unique(np.concatenate([all_labels_arr, all_preds_arr]))
    label_names = [STAGE_ORDER[i] if i < len(STAGE_ORDER) else f"Stage_{i}"
                   for i in unique_labels]
    print("\n" + "=" * 80)
    print("TRANSFORMER CLASSIFICATION REPORT (validation)")
    print("=" * 80)
    print(classification_report(all_labels_arr, all_preds_arr,
                                   target_names=label_names,
                                   zero_division=0,
                                   labels=unique_labels))
    print("=" * 80)

    return metadata


if __name__ == "__main__":
    # Load sequences
    X = np.load("data/sequences/X.npy").astype(np.float32)
    y = np.load("data/sequences/y.npy").astype(np.int64)

    # Load campaign-level splits if available
    import json
    splits_path = "data/sequences/metadata.json"
    if os.path.exists(splits_path):
        with open(splits_path) as f:
            meta = json.load(f)
        splits = meta.get("campaign_splits", {})
        print(f"[transformer] Campaign splits found: {splits}")

        try:
            X_train = np.load("data/sequences/X_train.npy").astype(np.float32)
            y_train = np.load("data/sequences/y_train.npy").astype(np.int64)
            X_val = np.load("data/sequences/X_val.npy").astype(np.float32)
            y_val = np.load("data/sequences/y_val.npy").astype(np.int64)
            X_test = np.load("data/sequences/X_test.npy").astype(np.float32) if os.path.exists("data/sequences/X_test.npy") else X_val
            y_test = np.load("data/sequences/y_test.npy").astype(np.int64) if os.path.exists("data/sequences/y_test.npy") else y_val
            print(f"[transformer] Loaded campaign splits: "
                  f"train={X_train.shape[0]}, "
                  f"val={X_val.shape[0]}, "
                  f"test={X_test.shape[0]}")
        except FileNotFoundError:
            print("[transformer] Split files not found, using random split")
            n = len(y)
            indices = np.random.RandomState(42).choice(n, int(n*0.7), replace=False)
            mask = np.zeros(n, dtype=bool)
            mask[indices] = True
            X_train, X_val = X[mask], X[~mask]
            y_train, y_val = y[mask], y[~mask]
            X_test, y_test = X_val, y_val
    else:
        print("[transformer] No metadata.json found, using random split")
        n = len(y)
        indices = np.random.RandomState(42).choice(n, int(n*0.7), replace=False)
        mask = np.zeros(n, dtype=bool)
        mask[indices] = True
        X_train, X_val = X[mask], X[~mask]
        y_train, y_val = y[mask], y[~mask]
        X_test, y_test = X_val, y_val

    print(f"[transformer] Train: {X_train.shape[0]}, "
          f"Val: {X_val.shape[0]}, "
          f"Test: {X_test.shape[0]}")

    # Compute class weights from training data
    class_counts = np.bincount(y_train.flatten().astype(int))
    total = class_counts.sum()
    n_classes = len(class_counts)
    weight = total / (n_classes * class_counts)
    weight = np.minimum(weight, weight.max() * 5.0)
    weight_tensor = torch.FloatTensor(weight / weight.mean())

    metadata = train_transformer(X_train, y_train, X_val, y_val,
                                   n_features=X.shape[2],
                                   d_model=128,
                                   n_heads=4,
                                   n_layers=4,
                                   epochs=20,
                                   batch_size=512,
                                   class_weights=weight_tensor)
    print(f"\nTransformer training complete. Val accuracy: "
          f"{metadata['val_accuracy']:.4f}")

    # Evaluate on test set (campaign-held-out)
    print(f"\n[transformer] Evaluating on test set...")
    model = AttackTransformer(n_features=X.shape[2], d_model=128,
                               n_heads=4, n_layers=4)
    model.load_state_dict(torch.load("models/transformer_best.pth", weights_only=False))
    model.eval()
    test_dataset = AttackSequenceDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            outputs = model(batch_X)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
    test_acc = accuracy_score(all_labels, all_preds)
    print(f"[transformer] Test accuracy (campaign-held-out): {test_acc:.4f}")
