"""Tests for evaluation functions."""

import math

import cv2
import numpy as np
import pytest
import torch

from src.evaluation.evaluate import get_ensemble_weight, predict_location


class TestGetEnsembleWeight:
    def test_average_mode(self):
        w = get_ensemble_weight(8, "average")
        assert torch.allclose(w, torch.ones(8) / 8)

    def test_weight_mode_sums_to_one(self):
        w = get_ensemble_weight(8, "weight")
        assert torch.isclose(w.sum(), torch.tensor(1.0))

    def test_weight_mode_symmetric(self):
        w = get_ensemble_weight(6, "weight")
        for i in range(3):
            assert torch.isclose(w[i], w[5 - i])

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="Invalid mode"):
            get_ensemble_weight(8, "invalid")


class TestPredictLocation:
    def test_empty_heatmap(self):
        heatmap = np.zeros((288, 512), dtype=np.uint8)
        x, y, w, h = predict_location(heatmap)
        assert (x, y, w, h) == (0, 0, 0, 0)

    def test_single_blob(self):
        heatmap = np.zeros((288, 512), dtype=np.uint8)
        # Draw a filled circle
        cv2.circle(heatmap, (256, 144), 10, 255, -1)
        x, y, w, h = predict_location(heatmap)
        # Should return bounding box near the circle
        assert w > 0 and h > 0
        cx, cy = x + w // 2, y + h // 2
        assert abs(cx - 256) <= 5
        assert abs(cy - 144) <= 5

    def test_largest_contour_selected(self):
        heatmap = np.zeros((288, 512), dtype=np.uint8)
        # Small blob
        cv2.circle(heatmap, (50, 50), 5, 255, -1)
        # Large blob
        cv2.circle(heatmap, (300, 200), 20, 255, -1)
        x, y, w, h = predict_location(heatmap)
        # Bounding box should correspond to the larger blob
        cx, cy = x + w // 2, y + h // 2
        assert abs(cx - 300) <= 5
        assert abs(cy - 200) <= 5

    def test_returns_four_values(self):
        heatmap = np.zeros((288, 512), dtype=np.uint8)
        result = predict_location(heatmap)
        assert len(result) == 4
