"""Tests for RF-DETR ball tracking inference pipeline."""

import numpy as np
import pytest
from collections import deque

from src.inference.ball_tracking import draw_ball_trajectory, BALL_THRESHOLD


class TestDrawBallTrajectory:
    def _blank_frame(self, h=480, w=640):
        return np.zeros((h, w, 3), dtype=np.uint8)

    def test_returns_same_shape(self):
        frame = self._blank_frame()
        traj = deque([None, (100, 100), None, (200, 200)])
        result = draw_ball_trajectory(frame, traj)
        assert result.shape == frame.shape

    def test_empty_trajectory_unchanged(self):
        """No dots to draw — result should still be the same shape."""
        frame = self._blank_frame()
        traj: deque = deque()
        result = draw_ball_trajectory(frame, traj)
        assert result.shape == frame.shape

    def test_none_positions_skipped(self):
        """None entries in trajectory should not crash."""
        frame = self._blank_frame()
        traj = deque([None, None, None])
        result = draw_ball_trajectory(frame, traj)
        assert result.shape == frame.shape

    def test_annotates_when_position_provided(self):
        """A valid position should produce non-zero pixels on a blank frame."""
        frame = self._blank_frame()
        traj = deque([(320, 240)])
        result = draw_ball_trajectory(frame, traj)
        # At least some pixels should have changed
        assert not np.array_equal(result, frame)

    def test_output_dtype_is_uint8(self):
        frame = self._blank_frame()
        traj = deque([(100, 100)])
        result = draw_ball_trajectory(frame, traj)
        assert result.dtype == np.uint8


class TestBallThreshold:
    def test_threshold_is_positive_float(self):
        assert isinstance(BALL_THRESHOLD, float)
        assert 0.0 < BALL_THRESHOLD < 1.0
