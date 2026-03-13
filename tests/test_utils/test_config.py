"""Tests for the configuration management system."""

import os
from pathlib import Path
from unittest.mock import patch

import yaml

from src.utils import config
from src.utils.config import (
    DatabaseConfig,
    ModelConfig,
    ServerConfig,
    Settings,
    TrainingConfig,
    get_settings,
    load_yaml_config,
)


class TestDefaultValues:
    """Verify all config classes have correct default values."""

    def test_server_defaults(self) -> None:
        cfg = ServerConfig()
        assert cfg.host == "0.0.0.0"
        assert cfg.port == 8000
        assert cfg.workers == 1
        assert cfg.reload is False

    def test_model_defaults(self) -> None:
        cfg = ModelConfig()
        assert cfg.model_dir == "./models"
        assert cfg.default_model == "hybrid"
        assert cfg.image_size == 512
        assert cfg.device == "auto"

    def test_training_defaults(self) -> None:
        cfg = TrainingConfig()
        assert cfg.lr == 0.0005
        assert cfg.batch_size == 32
        assert cfg.epochs == 50
        assert cfg.focal_alpha == 0.25
        assert cfg.focal_gamma == 2.0
        assert cfg.boundary_weight == 0.3
        assert cfg.patience == 10

    def test_database_defaults(self) -> None:
        cfg = DatabaseConfig()
        assert cfg.url == "sqlite:///./predictions.db"

    def test_settings_defaults(self) -> None:
        settings = Settings()
        assert settings.server.host == "0.0.0.0"
        assert settings.model.default_model == "hybrid"
        assert settings.training.lr == 0.0005
        assert settings.database.url == "sqlite:///./predictions.db"
        assert settings.mlflow_tracking_uri == "./mlruns"
        assert settings.api_key == "your-api-key-here"


class TestEnvOverride:
    """Verify environment variables override defaults."""

    def test_server_env_override(self) -> None:
        with patch.dict(os.environ, {"SERVER_PORT": "9000", "SERVER_HOST": "127.0.0.1"}):
            cfg = ServerConfig()
            assert cfg.port == 9000
            assert cfg.host == "127.0.0.1"

    def test_model_env_override(self) -> None:
        with patch.dict(os.environ, {"MODEL_DEFAULT_MODEL": "cnn", "MODEL_IMAGE_SIZE": "256"}):
            cfg = ModelConfig()
            assert cfg.default_model == "cnn"
            assert cfg.image_size == 256

    def test_training_env_override(self) -> None:
        with patch.dict(os.environ, {"TRAINING_LR": "0.001", "TRAINING_BATCH_SIZE": "16"}):
            cfg = TrainingConfig()
            assert cfg.lr == 0.001
            assert cfg.batch_size == 16

    def test_database_env_override(self) -> None:
        with patch.dict(os.environ, {"DATABASE_URL": "postgresql://localhost/medseg"}):
            cfg = DatabaseConfig()
            assert cfg.url == "postgresql://localhost/medseg"


class TestDeviceAutoDetection:
    """Verify get_device resolves correctly."""

    def test_explicit_cpu(self) -> None:
        cfg = ModelConfig(device="cpu")
        assert cfg.get_device() == "cpu"

    def test_explicit_cuda(self) -> None:
        cfg = ModelConfig(device="cuda")
        assert cfg.get_device() == "cuda"

    def test_auto_resolves_to_cpu_or_cuda(self) -> None:
        cfg = ModelConfig(device="auto")
        result = cfg.get_device()
        assert result in ("cpu", "cuda")

    def test_auto_without_cuda(self) -> None:
        with patch("torch.cuda.is_available", return_value=False):
            cfg = ModelConfig(device="auto")
            assert cfg.get_device() == "cpu"

    def test_auto_with_cuda(self) -> None:
        with patch("torch.cuda.is_available", return_value=True):
            cfg = ModelConfig(device="auto")
            assert cfg.get_device() == "cuda"


class TestSingleton:
    """Verify get_settings returns the same instance."""

    def setup_method(self) -> None:
        config._settings_instance = None

    def teardown_method(self) -> None:
        config._settings_instance = None

    def test_singleton_returns_same_instance(self) -> None:
        first = get_settings()
        second = get_settings()
        assert first is second

    def test_singleton_resets_when_cleared(self) -> None:
        first = get_settings()
        config._settings_instance = None
        second = get_settings()
        assert first is not second


class TestLoadYamlConfig:
    """Verify YAML config loading."""

    def test_load_valid_yaml(self, tmp_path: Path) -> None:
        yaml_file = tmp_path / "config.yaml"
        data = {"server": {"host": "127.0.0.1", "port": 9000}}
        yaml_file.write_text(yaml.dump(data))

        result = load_yaml_config(yaml_file)
        assert result["server"]["host"] == "127.0.0.1"
        assert result["server"]["port"] == 9000

    def test_load_missing_file(self) -> None:
        result = load_yaml_config("/nonexistent/path.yaml")
        assert result == {}

    def test_load_empty_yaml(self, tmp_path: Path) -> None:
        yaml_file = tmp_path / "empty.yaml"
        yaml_file.write_text("")

        result = load_yaml_config(yaml_file)
        assert result == {}
