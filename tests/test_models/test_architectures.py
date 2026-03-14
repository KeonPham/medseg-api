"""Tests for model architectures: CNN, ViT, and Hybrid."""

import pytest
import torch
import torch.nn as nn

from src.models.architectures import get_model
from src.models.architectures.cnn_model import CNNLungSegmentation
from src.models.architectures.hybrid_model import HybridLungSegmentation
from src.models.architectures.vit_model import ViTLungSegmentation

# Use small image size for faster tests
TEST_IMG_SIZE = 512
BATCH_SIZE = 1


@pytest.fixture(scope="module")
def cnn_model() -> CNNLungSegmentation:
    """Create CNN model (no pretrained weights for speed)."""
    return CNNLungSegmentation(img_size=TEST_IMG_SIZE, pretrained=False)


@pytest.fixture(scope="module")
def vit_model() -> ViTLungSegmentation:
    """Create ViT model (no pretrained weights for speed)."""
    return ViTLungSegmentation(img_size=TEST_IMG_SIZE, pretrained=False)


@pytest.fixture(scope="module")
def hybrid_model() -> HybridLungSegmentation:
    """Create Hybrid model (no pretrained weights for speed)."""
    return HybridLungSegmentation(img_size=TEST_IMG_SIZE, pretrained=False)


@pytest.fixture(scope="module")
def dummy_input() -> torch.Tensor:
    """Create a dummy input tensor."""
    return torch.randn(BATCH_SIZE, 3, TEST_IMG_SIZE, TEST_IMG_SIZE)


class TestCNNModel:
    """Tests for the CNN-only model."""

    def test_instantiation(self, cnn_model: CNNLungSegmentation) -> None:
        assert isinstance(cnn_model, nn.Module)
        assert hasattr(cnn_model, "cnn_encoder")
        assert hasattr(cnn_model, "decoder")

    def test_forward(self, cnn_model: CNNLungSegmentation, dummy_input: torch.Tensor) -> None:
        output = cnn_model(dummy_input)
        assert output.shape == (BATCH_SIZE, 1, TEST_IMG_SIZE, TEST_IMG_SIZE)

    def test_output_dtype(self, cnn_model: CNNLungSegmentation, dummy_input: torch.Tensor) -> None:
        output = cnn_model(dummy_input)
        assert output.dtype == torch.float32


class TestViTModel:
    """Tests for the ViT-only model."""

    def test_instantiation(self, vit_model: ViTLungSegmentation) -> None:
        assert isinstance(vit_model, nn.Module)
        assert hasattr(vit_model, "vit")
        assert hasattr(vit_model, "feature_proj")
        assert hasattr(vit_model, "decoder")

    def test_forward(self, vit_model: ViTLungSegmentation, dummy_input: torch.Tensor) -> None:
        output = vit_model(dummy_input)
        assert output.shape == (BATCH_SIZE, 1, TEST_IMG_SIZE, TEST_IMG_SIZE)

    def test_output_dtype(self, vit_model: ViTLungSegmentation, dummy_input: torch.Tensor) -> None:
        output = vit_model(dummy_input)
        assert output.dtype == torch.float32


class TestHybridModel:
    """Tests for the Hybrid CNN-ViT model."""

    def test_instantiation(self, hybrid_model: HybridLungSegmentation) -> None:
        assert isinstance(hybrid_model, nn.Module)
        assert hasattr(hybrid_model, "cnn_encoder")
        assert hasattr(hybrid_model, "vit_encoder")
        assert hasattr(hybrid_model, "cross_attention")
        assert hasattr(hybrid_model, "decoder")

    def test_forward(self, hybrid_model: HybridLungSegmentation, dummy_input: torch.Tensor) -> None:
        output = hybrid_model(dummy_input)
        assert output.shape == (BATCH_SIZE, 1, TEST_IMG_SIZE, TEST_IMG_SIZE)

    def test_output_dtype(
        self, hybrid_model: HybridLungSegmentation, dummy_input: torch.Tensor
    ) -> None:
        output = hybrid_model(dummy_input)
        assert output.dtype == torch.float32


class TestOutputShape:
    """Verify all models produce the correct output shape."""

    @pytest.mark.parametrize("model_name", ["cnn", "vit", "hybrid"])
    def test_output_shape(self, model_name: str) -> None:
        model = get_model(model_name, img_size=TEST_IMG_SIZE, pretrained=False)
        x = torch.randn(BATCH_SIZE, 3, TEST_IMG_SIZE, TEST_IMG_SIZE)
        output = model(x)
        assert output.shape == (BATCH_SIZE, 1, TEST_IMG_SIZE, TEST_IMG_SIZE)


class TestFactory:
    """Tests for the get_model factory function."""

    def test_get_cnn(self) -> None:
        model = get_model("cnn", pretrained=False)
        assert isinstance(model, CNNLungSegmentation)

    def test_get_vit(self) -> None:
        model = get_model("vit", pretrained=False)
        assert isinstance(model, ViTLungSegmentation)

    def test_get_hybrid(self) -> None:
        model = get_model("hybrid", pretrained=False)
        assert isinstance(model, HybridLungSegmentation)

    def test_invalid_name(self) -> None:
        with pytest.raises(ValueError, match="Unknown model"):
            get_model("nonexistent")


class TestParamCount:
    """Verify each model reports reasonable parameter counts."""

    def test_cnn_param_count(self, cnn_model: CNNLungSegmentation) -> None:
        count = cnn_model.get_param_count()
        assert 10_000_000 < count < 25_000_000  # ~15M expected

    def test_vit_param_count(self, vit_model: ViTLungSegmentation) -> None:
        count = vit_model.get_param_count()
        assert 5_000_000 < count < 15_000_000  # ~8M expected

    def test_hybrid_param_count(self, hybrid_model: HybridLungSegmentation) -> None:
        count = hybrid_model.get_param_count()
        assert 15_000_000 < count < 30_000_000  # ~22M with both encoders

    def test_parameter_info(self, hybrid_model: HybridLungSegmentation) -> None:
        info = hybrid_model.get_parameter_info()
        assert "cnn_encoder" in info
        assert "vit_encoder" in info
        assert "cross_attention" in info
        assert "decoder" in info
        assert info["total"] == hybrid_model.get_param_count()
