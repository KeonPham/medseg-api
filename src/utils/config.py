"""Application configuration loading from environment variables and YAML."""

import logging
from pathlib import Path

import yaml
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)

_settings_instance: "Settings | None" = None


class ServerConfig(BaseSettings):
    """Server configuration."""

    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    reload: bool = False

    model_config = {"env_prefix": "SERVER_"}


class ModelConfig(BaseSettings):
    """Model serving configuration."""

    model_dir: str = "./models"
    default_model: str = "hybrid"
    image_size: int = 512
    device: str = "auto"

    model_config = {"env_prefix": "MODEL_"}

    def get_device(self) -> str:
        """Resolve the compute device.

        Returns:
            'cuda' if available and device is 'auto', otherwise 'cpu' or the explicit value.
        """
        if self.device == "auto":
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device


class TrainingConfig(BaseSettings):
    """Training hyperparameter configuration."""

    lr: float = 0.0005
    batch_size: int = 32
    epochs: int = 50
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    boundary_weight: float = 0.3
    patience: int = 10

    model_config = {"env_prefix": "TRAINING_"}


class DatabaseConfig(BaseSettings):
    """Database connection configuration."""

    url: str = "sqlite:///./predictions.db"

    model_config = {"env_prefix": "DATABASE_"}


class Settings(BaseSettings):
    """Top-level application settings composed of sub-configurations."""

    server: ServerConfig = ServerConfig()
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    database: DatabaseConfig = DatabaseConfig()
    mlflow_tracking_uri: str = "./mlruns"
    api_key: str = "your-api-key-here"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


def load_yaml_config(config_path: str | Path) -> dict:
    """Load configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Dictionary of configuration values.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        logger.warning("Config file not found: %s", config_path)
        return {}

    with open(config_path) as f:
        config = yaml.safe_load(f)

    logger.info("Loaded config from %s", config_path)
    return config or {}


def get_settings() -> Settings:
    """Return the application settings singleton.

    Returns:
        Settings instance, created on first call and cached thereafter.
    """
    global _settings_instance  # noqa: PLW0603
    if _settings_instance is None:
        _settings_instance = Settings()
    return _settings_instance
