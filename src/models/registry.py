"""Model registry for managing, versioning, and serving segmentation models."""

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import torch
import yaml

from src.models.architectures import get_model

logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """Metadata for a registered model version."""

    name: str
    version: str
    architecture: str
    metrics: dict = field(default_factory=dict)
    file_path: str = ""
    loaded_at: datetime | None = None
    is_active: bool = False
    param_count: int = 0


class ModelRegistry:
    """Thread-safe registry for loading and caching segmentation models.

    Reads model definitions from a YAML config and manages the lifecycle
    of model instances: creation, weight loading, caching, and lookup.
    """

    def __init__(self, config_path: str = "configs/model_registry.yaml") -> None:
        """Initialize the registry from a YAML configuration file.

        Args:
            config_path: Path to the model registry YAML config.
        """
        self._lock = threading.Lock()
        self._cache: dict[str, torch.nn.Module] = {}
        self._models: dict[str, ModelInfo] = {}
        self._default_model: str = "hybrid"

        config_path = Path(config_path)
        if config_path.exists():
            with open(config_path) as f:
                config = yaml.safe_load(f) or {}
            self._default_model = config.get("default_model", "hybrid")
            self._parse_config(config)
            logger.info("ModelRegistry loaded %d models from %s", len(self._models), config_path)
        else:
            logger.warning("Registry config not found: %s", config_path)

    def _parse_config(self, config: dict) -> None:
        """Parse the YAML config into ModelInfo entries.

        Args:
            config: Parsed YAML dictionary.
        """
        models_cfg = config.get("models", {})
        for model_name, model_def in models_cfg.items():
            architecture = model_def.get("architecture", model_name)
            versions = model_def.get("versions", {})
            for version_id, version_def in versions.items():
                key = f"{model_name}:{version_id}"
                self._models[key] = ModelInfo(
                    name=model_name,
                    version=version_id,
                    architecture=architecture,
                    metrics=version_def.get("metrics", {}),
                    file_path=version_def.get("path", ""),
                )
            # Also register a "latest" alias pointing to the last version
            if versions:
                latest_version = list(versions.keys())[-1]
                latest_def = versions[latest_version]
                self._models[f"{model_name}:latest"] = ModelInfo(
                    name=model_name,
                    version=latest_version,
                    architecture=architecture,
                    metrics=latest_def.get("metrics", {}),
                    file_path=latest_def.get("path", ""),
                )

    def load_model(
        self, name: str, version: str = "latest", device: str = "cpu"
    ) -> torch.nn.Module:
        """Load a model by name and version, caching the result.

        Args:
            name: Model name (e.g. 'hybrid', 'cnn_only').
            version: Version tag (e.g. 'v1') or 'latest'.
            device: Device to load the model onto.

        Returns:
            The loaded model in eval mode.

        Raises:
            KeyError: If the model name/version is not registered.
            FileNotFoundError: If the weight file does not exist.
        """
        key = f"{name}:{version}"

        with self._lock:
            if key in self._cache:
                logger.info("Returning cached model: %s", key)
                return self._cache[key]

            info = self._models.get(key)
            if info is None:
                raise KeyError(f"Model '{key}' not found in registry")

            weight_path = Path(info.file_path)
            if not weight_path.exists():
                raise FileNotFoundError(f"Weight file not found: {weight_path}")

            model = get_model(info.architecture, pretrained=False)
            state_dict = torch.load(weight_path, map_location=device, weights_only=True)
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()

            info.loaded_at = datetime.now()
            info.is_active = True
            info.param_count = sum(p.numel() for p in model.parameters())

            self._cache[key] = model

        logger.info(
            "Loaded model %s (%s params) on %s",
            key,
            f"{info.param_count:,}",
            device,
        )
        return model

    def get_model(self, name: str, device: str = "cpu") -> torch.nn.Module:
        """Get a model by name, loading it if not already cached.

        Thread-safe convenience method that defaults to the latest version.

        Args:
            name: Model name (e.g. 'hybrid', 'cnn_only').
            device: Device to load the model onto if not cached.

        Returns:
            The model in eval mode.
        """
        key = f"{name}:latest"
        with self._lock:
            if key in self._cache:
                return self._cache[key]
        return self.load_model(name, version="latest", device=device)

    def list_models(self) -> list[ModelInfo]:
        """Return metadata for all registered model versions.

        Returns:
            List of ModelInfo dataclass instances (excludes 'latest' aliases).
        """
        return [info for key, info in self._models.items() if not key.endswith(":latest")]

    def get_model_info(self, name: str, version: str = "latest") -> ModelInfo | None:
        """Return metadata for a specific model.

        Args:
            name: Model name.
            version: Version tag or 'latest'.

        Returns:
            ModelInfo if found, otherwise None.
        """
        return self._models.get(f"{name}:{version}")

    @property
    def default_model(self) -> str:
        """The default model name from config."""
        return self._default_model
