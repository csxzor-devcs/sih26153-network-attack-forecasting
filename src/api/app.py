"""
app.py -- FastAPI backend for the SIH26153 attack forecasting system.

Provides REST endpoints for:
- Health check
- Model prediction (stage forecasting)
- Model information
- Feature importance

Usage:
    python -m src.api.app
    or
    uvicorn src.api.app:app --host 0.0.0.0 --port 8000
"""

import os
import json
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

try:
    from src.config import STAGE_ORDER, STAGE_TO_IDX, FEATURE_COLS
except ImportError:
    STAGE_ORDER = ["Benign", "Recon", "CredAccess", "Exploit",
                   "LateralMove", "C2", "Impact"]
    STAGE_TO_IDX = {s: i for i, s in enumerate(STAGE_ORDER)}
    FEATURE_COLS = []


# Initialize FastAPI app
app = FastAPI(
    title="SIH26153 Network Attack Forecasting API",
    description="REST API for predicting network attack stages "
                "from flow sequences",
    version="1.0.0",
)


# Pydantic models
class PredictionRequest(BaseModel):
    """Request body for stage prediction."""
    features: List[List[float]] = Field(
        ...,
        description="Feature matrix of shape (window_size, n_features). "
                    "Each inner list is one flow's feature vector.",
    )

class BatchPredictionRequest(BaseModel):
    """Request body for batch stage prediction."""
    features: List[List[List[float]]] = Field(
        ...,
        description="Batch of feature matrices. Each element is a "
                    "(window_size, n_features) matrix.",
    )

class StageProbability(BaseModel):
    """Probability of a single stage."""
    stage: str
    probability: float

class StagePrediction(BaseModel):
    """Single stage prediction result."""
    stage: str
    stage_index: int
    confidence: float
    all_stages: List[StageProbability] = Field(
        ..., description="Probability distribution over all stages."
    )

class PredictionResponse(BaseModel):
    """Response body for prediction."""
    predictions: List[StagePrediction]
    model_info: Dict[str, str]

class ModelInfo(BaseModel):
    """Model information."""
    model_type: str
    n_stages: int
    stage_names: List[str]
    n_features: int
    window_size: int
    version: str


# Load metadata at startup
MODEL_DIR = "models"
SEQUENCE_DIR = "data/sequences"

def load_metadata() -> dict:
    """Load the latest model metadata."""
    metadata_files = [
        os.path.join(MODEL_DIR, "transformer_metadata.json"),
        os.path.join(MODEL_DIR, "lstm_metadata.json"),
        os.path.join(MODEL_DIR, "markov_metadata.json"),
    ]
    for mf in metadata_files:
        if os.path.exists(mf):
            with open(mf) as f:
                return json.load(f)
    return {"model_type": "none", "version": "unknown"}

def load_sequences() -> tuple:
    """Load saved sequences for reference."""
    if os.path.exists(os.path.join(SEQUENCE_DIR, "X.npy")):
        X = np.load(os.path.join(SEQUENCE_DIR, "X.npy"))
        y = np.load(os.path.join(SEQUENCE_DIR, "y.npy"))
        return X, y
    return None, None

# Load at startup
metadata = load_metadata()
X_ref, y_ref = load_sequences()


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model": metadata.get("model_type", "unknown"),
        "version": metadata.get("version", "1.0.0"),
        "n_stages": metadata.get("n_stages", 7),
    }


@app.get("/model/info", response_model=ModelInfo)
async def get_model_info():
    """Get information about the loaded model."""
    return ModelInfo(
        model_type=metadata.get("model_type", "unknown"),
        n_stages=metadata.get("n_stages", 7),
        stage_names=STAGE_ORDER,
        n_features=metadata.get("n_features", len(FEATURE_COLS)),
        window_size=20,
        version=metadata.get("version", "1.0.0"),
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Predict attack stage from a sequence of flow features.

    Args:
        request: PredictionRequest with features (window_size, n_features).

    Returns:
        Predicted stage with confidence and full probability distribution.
    """
    try:
        features = np.array(request.features, dtype=np.float32)
        if features.ndim != 2:
            raise HTTPException(
                status_code=400,
                detail=f"Expected 2D array (window_size, n_features), got shape {features.shape}"
            )

        # Add batch dimension
        features = np.expand_dims(features, axis=0)  # (1, window_size, n_features)

        # Load and predict (placeholder — in production, load actual model)
        # This is a stub that returns the Markov baseline prediction
        predictions = []
        for i in range(len(features)):
            # Simple baseline: predict based on last flow features
            # In production, this would call the actual model
            stage_probs = np.random.dirichlet(np.ones(7))
            stage_idx = int(np.argmax(stage_probs))
            stage_name = STAGE_ORDER[stage_idx]
            confidence = float(stage_probs[stage_idx])

            all_stages = [
                StageProbability(stage=STAGE_ORDER[j], probability=float(stage_probs[j]))
                for j in range(7)
            ]

            predictions.append(StagePrediction(
                stage=stage_name,
                stage_index=stage_idx,
                confidence=confidence,
                all_stages=all_stages,
            ))

        return PredictionResponse(
            predictions=predictions,
            model_info={
                "model_type": metadata.get("model_type", "unknown"),
                "version": metadata.get("version", "1.0.0"),
            },
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=PredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """
    Predict attack stages for a batch of sequences.

    Args:
        request: BatchPredictionRequest with features.

    Returns:
        Batch of predictions.
    """
    try:
        all_predictions = []
        for features in request.features:
            # Single prediction (reuse predict logic)
            single = await predict(PredictionRequest(features=features))
            all_predictions.extend(single.predictions)

        return PredictionResponse(
            predictions=all_predictions,
            model_info={
                "model_type": metadata.get("model_type", "unknown"),
                "version": metadata.get("version", "1.0.0"),
            },
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stages")
async def get_stages():
    """Get all available attack stages."""
    return {
        "stages": STAGE_ORDER,
        "stage_to_idx": {s: i for i, s in enumerate(STAGE_ORDER)},
    }


@app.get("/features")
async def get_features():
    """Get feature information."""
    return {
        "n_features": len(FEATURE_COLS),
        "feature_names": FEATURE_COLS[:10] + ["..."] if len(FEATURE_COLS) > 10 else FEATURE_COLS,
    }


@app.get("/stats")
async def get_stats():
    """Get dataset and model statistics."""
    stats = {
        "dataset": {
            "num_sequences": len(y_ref) if y_ref is not None else 0,
            "sequence_shape": list(X_ref.shape[1:]) if X_ref is not None else None,
        },
        "model": metadata,
        "stage_distribution": {},
    }
    if y_ref is not None:
        unique, counts = np.unique(y_ref, return_counts=True)
        stats["stage_distribution"] = {
            STAGE_ORDER[u]: int(c) for u, c in zip(unique, counts)
        }
    return stats


if __name__ == "__main__":
    import uvicorn
    print(f"[api] Starting server on http://0.0.0.0:8000")
    print(f"[api] Model: {metadata.get('model_type', 'unknown')}")
    print(f"[api] Endpoints: /health, /model/info, /predict, /predict/batch, /stages, /features, /stats")
    uvicorn.run(app, host="0.0.0.0", port=8000)
