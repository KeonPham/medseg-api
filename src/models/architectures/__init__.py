"""Neural network architecture definitions for lung segmentation."""

import torch.nn as nn

from src.models.architectures.cnn_model import CNNLungSegmentation
from src.models.architectures.hybrid_model import HybridLungSegmentation
from src.models.architectures.vit_model import ViTLungSegmentation

_MODEL_MAP: dict[str, type[nn.Module]] = {
    "cnn": CNNLungSegmentation,
    "vit": ViTLungSegmentation,
    "hybrid": HybridLungSegmentation,
}


def get_model(name: str, **kwargs: object) -> nn.Module:
    """Factory function to create a model by name.

    Args:
        name: Model variant identifier ('cnn', 'vit', or 'hybrid').
        **kwargs: Forwarded to the model constructor (e.g. img_size, pretrained).

    Returns:
        Instantiated model.

    Raises:
        ValueError: If the model name is not recognized.
    """
    if name not in _MODEL_MAP:
        available = ", ".join(sorted(_MODEL_MAP.keys()))
        raise ValueError(f"Unknown model '{name}'. Available: {available}")
    return _MODEL_MAP[name](**kwargs)
