"""Tests for loss functions."""

import torch

from src.training.losses import BoundaryLoss, CombinedLoss, DiceLoss, FocalLoss


class TestFocalLoss:
    def test_shape(self) -> None:
        loss_fn = FocalLoss()
        pred = torch.randn(2, 1, 64, 64)
        target = torch.randint(0, 2, (2, 1, 64, 64)).float()
        loss = loss_fn(pred, target)
        assert loss.shape == ()
        assert loss.item() >= 0

    def test_perfect_prediction_low_loss(self) -> None:
        loss_fn = FocalLoss()
        # High logits where target=1, low where target=0
        target = torch.ones(1, 1, 16, 16)
        pred = torch.full((1, 1, 16, 16), 10.0)
        loss = loss_fn(pred, target)
        assert loss.item() < 0.01

    def test_gradient_flows(self) -> None:
        loss_fn = FocalLoss()
        pred = torch.randn(1, 1, 16, 16, requires_grad=True)
        target = torch.randint(0, 2, (1, 1, 16, 16)).float()
        loss = loss_fn(pred, target)
        loss.backward()
        assert pred.grad is not None


class TestDiceLoss:
    def test_range(self) -> None:
        loss_fn = DiceLoss()
        pred = torch.sigmoid(torch.randn(2, 1, 32, 32))
        target = torch.randint(0, 2, (2, 1, 32, 32)).float()
        loss = loss_fn(pred, target)
        assert 0.0 <= loss.item() <= 1.0

    def test_perfect_match_near_zero(self) -> None:
        loss_fn = DiceLoss()
        target = torch.ones(1, 1, 16, 16)
        pred = torch.ones(1, 1, 16, 16)
        loss = loss_fn(pred, target)
        assert loss.item() < 0.01

    def test_no_overlap_near_one(self) -> None:
        loss_fn = DiceLoss()
        pred = torch.zeros(1, 1, 16, 16)
        target = torch.ones(1, 1, 16, 16)
        loss = loss_fn(pred, target)
        assert loss.item() > 0.9


class TestBoundaryLoss:
    def test_shape(self) -> None:
        loss_fn = BoundaryLoss(weight=0.3)
        pred = torch.sigmoid(torch.randn(2, 1, 32, 32))
        target = torch.randint(0, 2, (2, 1, 32, 32)).float()
        loss = loss_fn(pred, target)
        assert loss.shape == ()
        assert loss.item() >= 0

    def test_zero_for_perfect_match(self) -> None:
        loss_fn = BoundaryLoss(weight=0.3)
        target = torch.ones(1, 1, 16, 16)
        pred = torch.ones(1, 1, 16, 16)
        loss = loss_fn(pred, target)
        assert loss.item() < 0.01


class TestCombinedLoss:
    def test_returns_scalar(self) -> None:
        loss_fn = CombinedLoss()
        pred = torch.randn(2, 1, 32, 32)
        target = torch.randint(0, 2, (2, 1, 32, 32)).float()
        loss = loss_fn(pred, target)
        assert loss.shape == ()
        assert loss.item() >= 0

    def test_gradient_flows(self) -> None:
        loss_fn = CombinedLoss()
        pred = torch.randn(1, 1, 16, 16, requires_grad=True)
        target = torch.randint(0, 2, (1, 1, 16, 16)).float()
        loss = loss_fn(pred, target)
        loss.backward()
        assert pred.grad is not None

    def test_includes_all_components(self) -> None:
        """Verify combined loss is greater than any single component."""
        pred = torch.randn(1, 1, 32, 32)
        target = torch.randint(0, 2, (1, 1, 32, 32)).float()

        combined = CombinedLoss()(pred, target).item()
        focal_only = FocalLoss()(pred, target).item()
        dice_only = DiceLoss()(torch.sigmoid(pred), target).item()

        assert combined >= focal_only
        assert combined >= dice_only
