"""
config_manager.py -- Centralized configuration management for SIH26153.

Loads settings from environment variables with sensible defaults,
enabling Docker/Kubernetes deployment without code changes.

Usage:
    from src.config_manager import Config
    config = Config()
    print(config.model_dir)
"""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Config:
    """Central application configuration loaded from environment variables."""

    # API settings
    api_host: str = field(default_factory=lambda: os.getenv("API_HOST", "0.0.0.0"))
    api_port: int = field(default_factory=lambda: int(os.getenv("API_PORT", "8000")))
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "info"))

    # Directory paths
    model_dir: str = field(default_factory=lambda: os.getenv("MODEL_DIR", "models"))
    sequence_dir: str = field(default_factory=lambda: os.getenv("SEQUENCE_DIR", "data/sequences"))
    processed_dir: str = field(default_factory=lambda: os.getenv("PROCESSED_DIR", "data/processed"))
    raw_dir: str = field(default_factory=lambda: os.getenv("RAW_DIR", "data/raw"))

    # Dataset settings
    max_rows: int = field(default_factory=lambda: int(os.getenv("MAX_ROWS", "100000")))
    n_campaigns: int = field(default_factory=lambda: int(os.getenv("N_CAMPAIGNS", "100")))
    window_size: int = field(default_factory=lambda: int(os.getenv("WINDOW_SIZE", "20")))
    eval_split: float = field(default_factory=lambda: float(os.getenv("EVALUATION_SPLIT", "0.8")))
    random_seed: int = field(default_factory=lambda: int(os.getenv("RANDOM_SEED", "42")))

    # Redis settings
    redis_host: str = field(default_factory=lambda: os.getenv("REDIS_HOST", "localhost"))
    redis_port: int = field(default_factory=lambda: int(os.getenv("REDIS_PORT", "6379")))

    # Model selection
    default_model: str = field(default_factory=lambda: os.getenv("DEFAULT_MODEL", "transformer"))

    # Frontend
    frontend_port: int = field(default_factory=lambda: int(os.getenv("FRONTEND_PORT", "3000")))
    react_app_api_url: str = field(default_factory=lambda: os.getenv("REACT_APP_API_URL", "http://localhost:8000"))

    # Derived properties
    @property
    def data_dir(self) -> str:
        """Return the base data directory."""
        return os.path.dirname(self.sequence_dir) if self.sequence_dir.startswith("data/") else "data"

    @property
    def use_redis(self) -> bool:
        """Whether Redis is available for caching."""
        return os.getenv("REDIS_ENABLED", "false").lower() == "true"

    def to_dict(self) -> dict:
        """Return configuration as a dictionary."""
        return {
            "api_host": self.api_host,
            "api_port": self.api_port,
            "log_level": self.log_level,
            "model_dir": self.model_dir,
            "sequence_dir": self.sequence_dir,
            "max_rows": self.max_rows,
            "n_campaigns": self.n_campaigns,
            "window_size": self.window_size,
            "default_model": self.default_model,
            "redis_host": self.redis_host,
            "redis_port": self.redis_port,
        }

    def __repr__(self) -> str:
        return f"Config(api_host={self.api_host}, api_port={self.api_port}, model_dir={self.model_dir})"


# Singleton instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Return the singleton Config instance."""
    global _config
    if _config is None:
        _config = Config()
    return _config


def reload_config() -> Config:
    """Re-create the Config singleton (useful for testing)."""
    global _config
    _config = Config()
    return _config
