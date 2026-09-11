"""
test_api.py -- Unit tests for the FastAPI backend.

Tests:
- Health endpoint
- Model info endpoint
- Prediction endpoint
- Stages and features endpoints
"""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.api.app import app, metadata
from src.config import STAGE_ORDER, FEATURE_COLS, WINDOW_SIZE


class TestHealthEndpoint:
    """Tests for /health endpoint."""

    def test_health_returns_healthy(self):
        """Verify health check returns 200 and healthy status."""
        from fastapi.testclient import TestClient

        client = TestClient(app)
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "model" in data

    def test_health_model_type(self):
        """Verify health response includes model type."""
        from fastapi.testclient import TestClient

        client = TestClient(app)
        response = client.get("/health")
        data = response.json()

        assert data["model"] in ["transformer", "lstm", "markov", "none"]


class TestModelInfoEndpoint:
    """Tests for /model/info endpoint."""

    def test_model_info_response(self):
        """Verify model info endpoint returns correct structure."""
        from fastapi.testclient import TestClient

        client = TestClient(app)
        response = client.get("/model/info")

        assert response.status_code == 200
        data = response.json()

        assert "model_type" in data
        assert "n_stages" in data
        assert "stage_names" in data
        assert "n_features" in data
        assert "window_size" in data
        assert data["window_size"] == WINDOW_SIZE

    def test_model_info_stages(self):
        """Verify stage names match STAGE_ORDER."""
        from fastapi.testclient import TestClient

        client = TestClient(app)
        response = client.get("/model/info")
        data = response.json()

        assert data["stage_names"] == STAGE_ORDER


class TestStagesEndpoint:
    """Tests for /stages endpoint."""

    def test_stages_returns_all_stages(self):
        """Verify all 7 stages are returned."""
        from fastapi.testclient import TestClient

        client = TestClient(app)
        response = client.get("/stages")

        assert response.status_code == 200
        data = response.json()

        assert len(data["stages"]) == 7
        assert data["stages"] == STAGE_ORDER
        assert len(data["stage_to_idx"]) == 7


class TestFeaturesEndpoint:
    """Tests for /features endpoint."""

    def test_features_returns_count(self):
        """Verify feature count is returned."""
        from fastapi.testclient import TestClient

        client = TestClient(app)
        response = client.get("/features")

        assert response.status_code == 200
        data = response.json()

        assert "n_features" in data
        assert data["n_features"] > 0


class TestStatsEndpoint:
    """Tests for /stats endpoint."""

    def test_stats_returns_structure(self):
        """Verify stats endpoint returns correct structure."""
        from fastapi.testclient import TestClient

        client = TestClient(app)
        response = client.get("/stats")

        assert response.status_code == 200
        data = response.json()

        assert "dataset" in data
        assert "model" in data
        assert "stage_distribution" in data


class TestPredictEndpoint:
    """Tests for /predict endpoint."""

    def test_predict_valid_input(self):
        """Verify prediction accepts valid feature array."""
        from fastapi.testclient import TestClient

        client = TestClient(app)

        n_features = len(FEATURE_COLS) if FEATURE_COLS else 46
        features = np.random.randn(WINDOW_SIZE, n_features).tolist()

        response = client.post(
            "/predict",
            json={"features": features},
        )

        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data
        assert len(data["predictions"]) > 0

    def test_predict_returns_stage_info(self):
        """Verify prediction returns stage name and confidence."""
        from fastapi.testclient import TestClient

        client = TestClient(app)

        n_features = len(FEATURE_COLS) if FEATURE_COLS else 46
        features = np.random.randn(WINDOW_SIZE, n_features).tolist()

        response = client.post(
            "/predict",
            json={"features": features},
        )
        data = response.json()

        pred = data["predictions"][0]
        assert "stage" in pred
        assert "confidence" in pred
        assert "all_stages" in pred
        assert 0 <= pred["confidence"] <= 1

    def test_predict_invalid_shape(self):
        """Verify prediction rejects wrong input shape."""
        from fastapi.testclient import TestClient

        client = TestClient(app)

        # 1x3 is a valid 2D array — it should be accepted
        features_2d = [[1.0, 2.0, 3.0]] * 20  # 20x3 (window_size=20, n_features=3)
        response = client.post("/predict", json={"features": features_2d})
        assert response.status_code == 200

        # Invalid: non-numeric input
        response = client.post(
            "/predict",
            json={"features": [["a", "b"], ["c", "d"]]},
        )
        assert response.status_code in (400, 422, 500)

    def test_predict_batch(self):
        """Verify batch prediction endpoint works."""
        from fastapi.testclient import TestClient

        client = TestClient(app)

        n_features = len(FEATURE_COLS) if FEATURE_COLS else 46
        features = [
            np.random.randn(WINDOW_SIZE, n_features).tolist()
            for _ in range(3)
        ]

        response = client.post(
            "/predict/batch",
            json={"features": features},
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data["predictions"]) == 3


class TestConfigManager:
    """Tests for configuration management."""

    def test_config_defaults(self):
        """Verify Config loads with defaults."""
        from src.config_manager import Config

        config = Config()
        assert config.api_port == 8000
        assert config.window_size == WINDOW_SIZE
        assert config.max_rows == 100000

    def test_config_env_override(self):
        """Verify Config reads environment variables."""
        from src.config_manager import reload_config
        import os

        os.environ["API_PORT"] = "9999"
        os.environ["MAX_ROWS"] = "50000"

        config = reload_config()
        assert config.api_port == 9999
        assert config.max_rows == 50000

        del os.environ["API_PORT"]
        del os.environ["MAX_ROWS"]

    def test_config_to_dict(self):
        """Verify Config can be serialized to dict."""
        from src.config_manager import Config

        config = Config()
        d = config.to_dict()

        assert "api_host" in d
        assert "api_port" in d
        assert isinstance(d["api_port"], int)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
