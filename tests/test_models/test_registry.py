"""Tests for the model registry system."""

import threading
from pathlib import Path

import pytest
import torch
import yaml

from src.models.registry import ModelInfo, ModelRegistry


@pytest.fixture()
def registry_config(tmp_path: Path) -> Path:
    """Create a temporary model registry config with dummy weight files."""
    # Create dummy weight files by saving a small state_dict
    dummy_sd = {"weight": torch.randn(2, 2)}

    models_dir = tmp_path / "models"
    for subdir in ("cnn_only", "vit_only", "hybrid"):
        d = models_dir / subdir
        d.mkdir(parents=True)
        torch.save(dummy_sd, d / f"lung_seg_{subdir.replace('_only', '')}_v1.pth")

    config = {
        "models": {
            "cnn_only": {
                "architecture": "cnn",
                "versions": {
                    "v1": {
                        "path": str(models_dir / "cnn_only" / "lung_seg_cnn_v1.pth"),
                        "metrics": {"dice": 0.9418, "iou": 0.8903},
                    },
                },
            },
            "vit_only": {
                "architecture": "vit",
                "versions": {
                    "v1": {
                        "path": str(models_dir / "vit_only" / "lung_seg_vit_v1.pth"),
                        "metrics": {"dice": 0.9518, "iou": 0.9082},
                    },
                },
            },
            "hybrid": {
                "architecture": "hybrid",
                "versions": {
                    "v1": {
                        "path": str(models_dir / "hybrid" / "lung_seg_hybrid_v1.pth"),
                        "metrics": {"dice": 0.9665, "iou": 0.9360},
                    },
                },
            },
        },
        "default_model": "hybrid",
    }

    config_path = tmp_path / "model_registry.yaml"
    config_path.write_text(yaml.dump(config))
    return config_path


@pytest.fixture()
def registry(registry_config: Path) -> ModelRegistry:
    """Create a ModelRegistry from the temp config."""
    return ModelRegistry(config_path=str(registry_config))


class TestListModels:
    """Test listing registered models."""

    def test_list_models_returns_all_three(self, registry: ModelRegistry) -> None:
        models = registry.list_models()
        assert len(models) == 3

    def test_list_models_returns_model_info(self, registry: ModelRegistry) -> None:
        models = registry.list_models()
        assert all(isinstance(m, ModelInfo) for m in models)

    def test_list_models_names(self, registry: ModelRegistry) -> None:
        names = {m.name for m in registry.list_models()}
        assert names == {"cnn_only", "vit_only", "hybrid"}


class TestGetModelInfo:
    """Test retrieving model metadata."""

    def test_get_hybrid_info(self, registry: ModelRegistry) -> None:
        info = registry.get_model_info("hybrid")
        assert info is not None
        assert info.name == "hybrid"
        assert info.architecture == "hybrid"
        assert info.version == "v1"
        assert info.metrics["dice"] == 0.9665

    def test_get_cnn_info(self, registry: ModelRegistry) -> None:
        info = registry.get_model_info("cnn_only", version="v1")
        assert info is not None
        assert info.architecture == "cnn"
        assert info.metrics["dice"] == 0.9418

    def test_get_nonexistent_returns_none(self, registry: ModelRegistry) -> None:
        info = registry.get_model_info("nonexistent")
        assert info is None

    def test_default_model(self, registry: ModelRegistry) -> None:
        assert registry.default_model == "hybrid"


class TestLoadModel:
    """Test model loading behavior."""

    def test_load_model_not_found_key(self, registry: ModelRegistry) -> None:
        with pytest.raises(KeyError, match="not found in registry"):
            registry.load_model("nonexistent")

    def test_load_model_missing_weights(self, tmp_path: Path) -> None:
        config = {
            "models": {
                "test_model": {
                    "architecture": "cnn",
                    "versions": {
                        "v1": {
                            "path": str(tmp_path / "does_not_exist.pth"),
                            "metrics": {},
                        },
                    },
                },
            },
        }
        config_path = tmp_path / "reg.yaml"
        config_path.write_text(yaml.dump(config))
        reg = ModelRegistry(config_path=str(config_path))
        with pytest.raises(FileNotFoundError, match="Weight file not found"):
            reg.load_model("test_model", version="v1")

    def test_load_real_model(self, registry_config: Path) -> None:
        """Load a CNN model using a real state_dict saved from the architecture."""
        # Create a proper state_dict from the actual CNN model
        from src.models.architectures.cnn_model import CNNLungSegmentation

        model = CNNLungSegmentation(pretrained=False)
        sd = model.state_dict()

        # Write it to the path the registry config points to
        config = yaml.safe_load(registry_config.read_text())
        cnn_path = Path(config["models"]["cnn_only"]["versions"]["v1"]["path"])
        torch.save(sd, cnn_path)

        reg = ModelRegistry(config_path=str(registry_config))
        loaded = reg.load_model("cnn_only", version="v1")

        assert isinstance(loaded, torch.nn.Module)
        assert not loaded.training  # eval mode

        # Info should be updated
        info = reg.get_model_info("cnn_only", version="v1")
        assert info is not None
        assert info.is_active
        assert info.loaded_at is not None
        assert info.param_count > 0

    def test_cached_model_returned(self, registry_config: Path) -> None:
        """Verify the same instance is returned on repeated loads."""
        from src.models.architectures.cnn_model import CNNLungSegmentation

        model = CNNLungSegmentation(pretrained=False)
        config = yaml.safe_load(registry_config.read_text())
        cnn_path = Path(config["models"]["cnn_only"]["versions"]["v1"]["path"])
        torch.save(model.state_dict(), cnn_path)

        reg = ModelRegistry(config_path=str(registry_config))
        first = reg.load_model("cnn_only")
        second = reg.load_model("cnn_only")
        assert first is second


class TestThreadSafety:
    """Verify concurrent access doesn't crash."""

    def test_concurrent_list_models(self, registry: ModelRegistry) -> None:
        errors: list[Exception] = []

        def worker() -> None:
            try:
                for _ in range(50):
                    models = registry.list_models()
                    assert len(models) == 3
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Thread errors: {errors}"

    def test_concurrent_get_info(self, registry: ModelRegistry) -> None:
        errors: list[Exception] = []

        def worker() -> None:
            try:
                for _ in range(50):
                    info = registry.get_model_info("hybrid")
                    assert info is not None
                    assert info.name == "hybrid"
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Thread errors: {errors}"

    def test_concurrent_load(self, registry_config: Path) -> None:
        """Multiple threads loading the same model should all get the same instance."""
        from src.models.architectures.cnn_model import CNNLungSegmentation

        model = CNNLungSegmentation(pretrained=False)
        config = yaml.safe_load(registry_config.read_text())
        cnn_path = Path(config["models"]["cnn_only"]["versions"]["v1"]["path"])
        torch.save(model.state_dict(), cnn_path)

        reg = ModelRegistry(config_path=str(registry_config))
        results: list[torch.nn.Module] = []
        errors: list[Exception] = []

        def worker() -> None:
            try:
                m = reg.get_model("cnn_only")
                results.append(m)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Thread errors: {errors}"
        assert len(results) == 8
        # All threads should get the same cached instance
        assert all(r is results[0] for r in results)


class TestMissingConfig:
    """Test behavior with missing or empty config."""

    def test_missing_config_file(self, tmp_path: Path) -> None:
        reg = ModelRegistry(config_path=str(tmp_path / "nonexistent.yaml"))
        assert reg.list_models() == []

    def test_empty_config(self, tmp_path: Path) -> None:
        config_path = tmp_path / "empty.yaml"
        config_path.write_text("")
        reg = ModelRegistry(config_path=str(config_path))
        assert reg.list_models() == []
