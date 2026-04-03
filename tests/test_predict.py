"""Tests for the predict function in src/inference/predict.py."""

import pytest
import torch

from src.inference.predict import predict
from src.utils.general import HEIGHT, WIDTH


class TestPredict:
    def test_from_heatmap_basic(self):
        """Test predict with heatmap input."""
        batch_size, seq_len = 1, 4
        indices = torch.zeros(batch_size, seq_len, 2)
        for f in range(seq_len):
            indices[0, f, 0] = 0  # match_id
            indices[0, f, 1] = f  # frame_id

        # Create heatmaps with a dot in different positions
        y_pred = torch.zeros(batch_size, seq_len, HEIGHT, WIDTH)
        # Put a bright spot at (100, 50) in frame 0
        y_pred[0, 0, 48:52, 98:102] = 1.0

        pred_dict = predict(indices, y_pred=y_pred, img_scaler=(1, 1))

        assert "Frame" in pred_dict
        assert "X" in pred_dict
        assert "Y" in pred_dict
        assert "Visibility" in pred_dict
        assert len(pred_dict["Frame"]) == seq_len

    def test_from_coordinates(self):
        """Test predict with coordinate input."""
        batch_size, seq_len = 1, 4
        indices = torch.zeros(batch_size, seq_len, 2)
        for f in range(seq_len):
            indices[0, f, 1] = f

        # Coordinates normalized to [0, 1]
        c_pred = torch.tensor([[[0.5, 0.5], [0.3, 0.3], [0.0, 0.0], [0.7, 0.7]]])

        pred_dict = predict(indices, c_pred=c_pred, img_scaler=(1, 1))
        assert len(pred_dict["Frame"]) == seq_len
        # Frame with (0, 0) coords should have visibility 0
        assert pred_dict["Visibility"][2] == 0

    def test_no_input_raises(self):
        """Should raise if neither y_pred nor c_pred provided."""
        indices = torch.zeros(1, 4, 2)
        for f in range(4):
            indices[0, f, 1] = f
        with pytest.raises(ValueError, match="Invalid input"):
            predict(indices)

    def test_img_scaler_applied(self):
        """Image scaler should scale output coordinates."""
        batch_size, seq_len = 1, 2
        indices = torch.zeros(batch_size, seq_len, 2)
        indices[0, 0, 1] = 0
        indices[0, 1, 1] = 1

        c_pred = torch.tensor([[[0.5, 0.5], [0.5, 0.5]]])
        w_scaler, h_scaler = 2.0, 2.0

        pred_dict = predict(indices, c_pred=c_pred, img_scaler=(w_scaler, h_scaler))
        # cx = 0.5 * WIDTH * 2.0, cy = 0.5 * HEIGHT * 2.0
        expected_x = int(0.5 * WIDTH * w_scaler)
        expected_y = int(0.5 * HEIGHT * h_scaler)
        assert pred_dict["X"][0] == expected_x
        assert pred_dict["Y"][0] == expected_y

    def test_sequential_batches(self):
        """Sequential non-overlapping batches should produce correct frame entries."""
        indices = torch.zeros(2, 4, 2)
        # Batch 0: frames 0, 1, 2, 3
        for f in range(4):
            indices[0, f, 1] = f
        # Batch 1: frames 4, 5, 6, 7
        for f in range(4):
            indices[1, f, 1] = f + 4

        c_pred = torch.rand(2, 4, 2) * 0.5 + 0.25  # away from zero
        pred_dict = predict(indices, c_pred=c_pred, img_scaler=(1, 1))

        assert len(pred_dict["Frame"]) == 8
        assert pred_dict["Frame"] == [0, 1, 2, 3, 4, 5, 6, 7]
