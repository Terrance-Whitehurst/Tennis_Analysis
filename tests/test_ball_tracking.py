"""Tests for ball tracking inference pipeline."""

import math

import numpy as np
import pytest
import torch

from src.inference.ball_tracking import (
    predict_ball_from_heatmap,
    generate_inpaint_mask,
    get_ensemble_weight,
    preprocess_sequence,
    HEIGHT,
    WIDTH,
)


class TestPredictBallFromHeatmap:
    def test_empty_heatmap_returns_zero(self):
        heatmap = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
        cx, cy = predict_ball_from_heatmap(heatmap)
        assert cx == 0
        assert cy == 0

    def test_single_dot(self):
        """A small white dot should be detected at approximately its center."""
        heatmap = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
        # Draw a small circle at (256, 144)
        center_x, center_y = 256, 144
        for dy in range(-3, 4):
            for dx in range(-3, 4):
                if dx * dx + dy * dy <= 9:
                    heatmap[center_y + dy, center_x + dx] = 255
        cx, cy = predict_ball_from_heatmap(heatmap)
        assert abs(cx - center_x) <= 2
        assert abs(cy - center_y) <= 2

    def test_picks_largest_contour(self):
        """When multiple blobs exist, returns center of the largest."""
        heatmap = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
        # Small blob at (50, 50)
        heatmap[48:52, 48:52] = 255
        # Larger blob at (200, 100)
        heatmap[90:110, 190:210] = 255
        cx, cy = predict_ball_from_heatmap(heatmap)
        # Should be near the larger blob center (200, 100)
        assert abs(cx - 200) <= 5
        assert abs(cy - 100) <= 5

    def test_returns_ints(self):
        heatmap = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
        heatmap[100:110, 100:110] = 255
        cx, cy = predict_ball_from_heatmap(heatmap)
        assert isinstance(cx, (int, np.integer))
        assert isinstance(cy, (int, np.integer))


class TestGenerateInpaintMask:
    def test_all_visible_no_mask(self):
        pred_dict = {
            "Y": [100, 100, 100, 100, 100],
            "Visibility": [1, 1, 1, 1, 1],
        }
        mask = generate_inpaint_mask(pred_dict, th_h=30.0)
        assert all(m == 0 for m in mask)

    def test_gap_in_middle_gets_masked(self):
        pred_dict = {
            "Y": [100, 100, 0, 0, 100, 100],
            "Visibility": [1, 1, 0, 0, 1, 1],
        }
        mask = generate_inpaint_mask(pred_dict, th_h=30.0)
        assert mask[2] == 1
        assert mask[3] == 1
        assert mask[0] == 0
        assert mask[1] == 0

    def test_gap_near_top_edge_not_masked(self):
        """If surrounding y values are below threshold, don't inpaint."""
        pred_dict = {
            "Y": [10, 10, 0, 0, 10, 10],
            "Visibility": [1, 1, 0, 0, 1, 1],
        }
        mask = generate_inpaint_mask(pred_dict, th_h=30.0)
        assert all(m == 0 for m in mask)

    def test_output_length_matches_input(self):
        n = 20
        pred_dict = {
            "Y": [100] * n,
            "Visibility": [1] * n,
        }
        mask = generate_inpaint_mask(pred_dict)
        assert len(mask) == n


class TestGetEnsembleWeight:
    def test_average_mode(self):
        w = get_ensemble_weight(8, "average")
        assert w.shape == (8,)
        assert torch.allclose(w, torch.ones(8) / 8)

    def test_weight_mode_sums_to_one(self):
        w = get_ensemble_weight(8, "weight")
        assert w.shape == (8,)
        assert torch.isclose(w.sum(), torch.tensor(1.0))

    def test_weight_mode_symmetric(self):
        w = get_ensemble_weight(8, "weight")
        for i in range(4):
            assert torch.isclose(w[i], w[7 - i])

    def test_weight_mode_center_heavier(self):
        """Center of the sequence should have higher weight."""
        w = get_ensemble_weight(8, "weight")
        assert w[3] > w[0]
        assert w[4] > w[0]

    def test_various_seq_lengths(self):
        for seq_len in [3, 4, 8, 16]:
            w = get_ensemble_weight(seq_len, "weight")
            assert w.shape == (seq_len,)
            assert torch.isclose(w.sum(), torch.tensor(1.0))


class TestPreprocessSequence:
    def _make_bgr_frame(self, h=480, w=640):
        return np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)

    def _make_median(self):
        return np.random.randint(0, 256, (3, HEIGHT, WIDTH), dtype=np.uint8)

    def test_concat_mode_shape(self):
        """8 frames + median => (8+1)*3 = 27 channels."""
        frames = [self._make_bgr_frame() for _ in range(8)]
        median = self._make_median()
        result = preprocess_sequence(frames, median, "concat")
        assert result.shape == (27, HEIGHT, WIDTH)

    def test_default_mode_shape(self):
        """Default bg_mode (empty string) => 8*3=24 channels."""
        frames = [self._make_bgr_frame() for _ in range(8)]
        median = self._make_median()
        result = preprocess_sequence(frames, median, "")
        assert result.shape == (24, HEIGHT, WIDTH)

    def test_output_normalized(self):
        """Values should be in [0, 1]."""
        frames = [self._make_bgr_frame() for _ in range(4)]
        median = self._make_median()
        result = preprocess_sequence(frames, median, "")
        assert result.min() >= 0.0
        assert result.max() <= 1.0
