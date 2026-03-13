"""Application configuration loading from environment variables and YAML."""

import logging
from pathlib import Path

import yaml
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_dir: Path = Path("./models")
    default_model: str = "hybrid"
    image_size: int = 512
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    database_url: str = "sqlite:///./predictions.db"
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
    """Create and return application settings instance.

    Returns:
        Settings instance with values from environment.
    """
    return Settings()
