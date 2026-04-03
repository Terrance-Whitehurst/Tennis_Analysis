"""Tests for TrackNet and InpaintNet model architectures."""

import pytest
import torch
from src.models.tracknet import (
    Conv2DBlock,
    Double2DConv,
    Triple2DConv,
    TrackNet,
    Conv1DBlock,
    InpaintNet,
)

HEIGHT, WIDTH = 288, 512


# ── Conv2DBlock ──────────────────────────────────────────────────────────────


class TestConv2DBlock:
    def test_output_shape(self):
        block = Conv2DBlock(3, 64)
        x = torch.randn(2, 3, 32, 32)
        out = block(x)
        assert out.shape == (2, 64, 32, 32)

    def test_preserves_spatial_dims(self):
        block = Conv2DBlock(16, 32)
        x = torch.randn(1, 16, HEIGHT, WIDTH)
        out = block(x)
        assert out.shape[2:] == (HEIGHT, WIDTH)

    def test_output_is_nonnegative(self):
        """ReLU should ensure non-negative outputs."""
        block = Conv2DBlock(3, 8)
        x = torch.randn(1, 3, 16, 16)
        out = block(x)
        assert (out >= 0).all()


# ── Double2DConv ─────────────────────────────────────────────────────────────


class TestDouble2DConv:
    def test_output_shape(self):
        block = Double2DConv(3, 64)
        x = torch.randn(2, 3, 32, 32)
        out = block(x)
        assert out.shape == (2, 64, 32, 32)


# ── Triple2DConv ─────────────────────────────────────────────────────────────


class TestTriple2DConv:
    def test_output_shape(self):
        block = Triple2DConv(64, 128)
        x = torch.randn(1, 64, 16, 16)
        out = block(x)
        assert out.shape == (1, 128, 16, 16)


# ── TrackNet ─────────────────────────────────────────────────────────────────


class TestTrackNet:
    @pytest.fixture
    def model(self):
        return TrackNet(in_dim=24, out_dim=8)

    def test_output_shape(self, model):
        x = torch.randn(1, 24, HEIGHT, WIDTH)
        out = model(x)
        assert out.shape == (1, 8, HEIGHT, WIDTH)

    def test_output_range_sigmoid(self, model):
        """Sigmoid output must be in [0, 1]."""
        x = torch.randn(1, 24, HEIGHT, WIDTH)
        out = model(x)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_batch_dimension(self, model):
        x = torch.randn(4, 24, HEIGHT, WIDTH)
        out = model(x)
        assert out.shape[0] == 4

    def test_default_rgb_config(self):
        """seq_len=8, bg_mode='' => in_dim=24 (8*3)."""
        model = TrackNet(in_dim=8 * 3, out_dim=8)
        x = torch.randn(1, 24, HEIGHT, WIDTH)
        out = model(x)
        assert out.shape == (1, 8, HEIGHT, WIDTH)

    def test_concat_bg_mode(self):
        """bg_mode='concat' => in_dim=(8+1)*3=27."""
        model = TrackNet(in_dim=27, out_dim=8)
        x = torch.randn(1, 27, HEIGHT, WIDTH)
        out = model(x)
        assert out.shape == (1, 8, HEIGHT, WIDTH)

    def test_subtract_bg_mode(self):
        """bg_mode='subtract' => in_dim=8 (grayscale diff frames)."""
        model = TrackNet(in_dim=8, out_dim=8)
        x = torch.randn(1, 8, HEIGHT, WIDTH)
        out = model(x)
        assert out.shape == (1, 8, HEIGHT, WIDTH)

    def test_subtract_concat_bg_mode(self):
        """bg_mode='subtract_concat' => in_dim=8*4=32."""
        model = TrackNet(in_dim=32, out_dim=8)
        x = torch.randn(1, 32, HEIGHT, WIDTH)
        out = model(x)
        assert out.shape == (1, 8, HEIGHT, WIDTH)

    def test_gradient_flow(self, model):
        """Verify gradients propagate through the full network."""
        x = torch.randn(1, 24, HEIGHT, WIDTH, requires_grad=True)
        out = model(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape


# ── Conv1DBlock ──────────────────────────────────────────────────────────────


class TestConv1DBlock:
    def test_output_shape(self):
        block = Conv1DBlock(3, 32)
        x = torch.randn(2, 3, 64)
        out = block(x)
        assert out.shape == (2, 32, 64)


# ── InpaintNet ───────────────────────────────────────────────────────────────


class TestInpaintNet:
    @pytest.fixture
    def model(self):
        return InpaintNet()

    def test_output_shape(self, model):
        seq_len = 16
        x = torch.randn(2, seq_len, 2)
        m = torch.ones(2, seq_len, 1)
        out = model(x, m)
        assert out.shape == (2, seq_len, 2)

    def test_output_range_sigmoid(self, model):
        x = torch.randn(1, 32, 2)
        m = torch.zeros(1, 32, 1)
        out = model(x, m)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_various_seq_lengths(self, model):
        for seq_len in [8, 16, 32, 64]:
            x = torch.randn(1, seq_len, 2)
            m = torch.zeros(1, seq_len, 1)
            out = model(x, m)
            assert out.shape == (1, seq_len, 2)

    def test_gradient_flow(self, model):
        x = torch.randn(1, 16, 2, requires_grad=True)
        m = torch.ones(1, 16, 1)
        out = model(x, m)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
