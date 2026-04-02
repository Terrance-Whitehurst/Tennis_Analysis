"""Tests for loss functions and evaluation metrics."""

import pytest
import torch
from src.utils.metric import WBCELoss, get_metric


class TestWBCELoss:
    def test_perfect_prediction_low_loss(self):
        """Loss should be near zero when prediction matches ground truth."""
        y = torch.ones(1, 1, 8, 8)
        y_pred = torch.ones(1, 1, 8, 8) * 0.999
        loss = WBCELoss(y_pred, y)
        assert loss.item() < 0.01

    def test_worst_prediction_high_loss(self):
        """Loss should be high when prediction is opposite of ground truth."""
        y = torch.ones(1, 1, 8, 8)
        y_pred = torch.ones(1, 1, 8, 8) * 0.001
        loss = WBCELoss(y_pred, y)
        assert loss.item() > 0.1

    def test_output_shape_reduced(self):
        y = torch.rand(4, 1, 16, 16)
        y_pred = torch.rand(4, 1, 16, 16)
        loss = WBCELoss(y_pred, y, reduce=True)
        assert loss.shape == ()

    def test_output_shape_unreduced(self):
        y = torch.rand(4, 1, 16, 16)
        y_pred = torch.rand(4, 1, 16, 16)
        loss = WBCELoss(y_pred, y, reduce=False)
        assert loss.shape == (4,)

    def test_loss_is_nonnegative(self):
        """WBCE loss should always be non-negative."""
        y = torch.rand(2, 1, 8, 8)
        y_pred = torch.rand(2, 1, 8, 8).clamp(0.01, 0.99)
        loss = WBCELoss(y_pred, y)
        assert loss.item() >= 0

    def test_symmetry_all_zeros(self):
        """All-zero ground truth with all-zero predictions should yield low loss."""
        y = torch.zeros(1, 1, 8, 8)
        y_pred = torch.zeros(1, 1, 8, 8) + 0.001
        loss = WBCELoss(y_pred, y)
        assert loss.item() < 0.01

    def test_gradient_flows(self):
        """Verify gradients propagate through the loss."""
        y = torch.rand(1, 1, 8, 8)
        y_pred = torch.rand(1, 1, 8, 8, requires_grad=True)
        loss = WBCELoss(y_pred, y)
        loss.backward()
        assert y_pred.grad is not None


class TestGetMetric:
    def test_perfect_classification(self):
        acc, prec, rec, f1, miss = get_metric(TP=50, TN=50, FP1=0, FP2=0, FN=0)
        assert acc == 1.0
        assert prec == 1.0
        assert rec == 1.0
        assert f1 == 1.0
        assert miss == 0.0

    def test_all_false_negatives(self):
        acc, prec, rec, f1, miss = get_metric(TP=0, TN=50, FP1=0, FP2=0, FN=50)
        assert acc == 0.5
        assert prec == 0  # no positive predictions
        assert rec == 0.0
        assert miss == 1.0

    def test_all_false_positives(self):
        acc, prec, rec, f1, miss = get_metric(TP=0, TN=0, FP1=25, FP2=25, FN=0)
        assert prec == 0.0
        assert acc == 0.0

    def test_all_zeros(self):
        """Edge case: no samples at all."""
        acc, prec, rec, f1, miss = get_metric(TP=0, TN=0, FP1=0, FP2=0, FN=0)
        assert acc == 0
        assert prec == 0
        assert rec == 0
        assert f1 == 0
        assert miss == 0

    def test_realistic_case(self):
        """TP=80, TN=10, FP1=5, FP2=3, FN=2 => check approximate values."""
        acc, prec, rec, f1, miss = get_metric(TP=80, TN=10, FP1=5, FP2=3, FN=2)
        assert acc == pytest.approx(90 / 100, rel=1e-6)
        assert prec == pytest.approx(80 / 88, rel=1e-6)
        assert rec == pytest.approx(80 / 82, rel=1e-6)
        expected_f1 = 2 * (80 / 88) * (80 / 82) / ((80 / 88) + (80 / 82))
        assert f1 == pytest.approx(expected_f1, rel=1e-6)
        assert miss == pytest.approx(2 / 82, rel=1e-6)

    def test_precision_counts_both_fp_types(self):
        """Precision denominator includes FP1 and FP2."""
        _, prec, _, _, _ = get_metric(TP=10, TN=0, FP1=5, FP2=5, FN=0)
        assert prec == pytest.approx(10 / 20)

    def test_miss_rate_complement_of_recall(self):
        """miss_rate = 1 - recall when TP + FN > 0."""
        _, _, rec, _, miss = get_metric(TP=30, TN=10, FP1=5, FP2=5, FN=10)
        assert miss == pytest.approx(1 - rec, rel=1e-6)
