"""Tests for utility functions in src/utils/general.py."""

import os
import tempfile

import numpy as np
import pytest
import torch

from src.utils.general import (
    get_model,
    list_dirs,
    to_img,
    to_img_format,
    HEIGHT,
    WIDTH,
    SIGMA,
    ResumeArgumentParser,
)
from src.models.tracknet import TrackNet, InpaintNet


class TestGetModel:
    def test_tracknet_default_bg(self):
        model = get_model("TrackNet", seq_len=8, bg_mode="")
        assert isinstance(model, TrackNet)
        # in_dim should be 8*3=24
        x = torch.randn(1, 24, HEIGHT, WIDTH)
        out = model(x)
        assert out.shape == (1, 8, HEIGHT, WIDTH)

    def test_tracknet_subtract(self):
        model = get_model("TrackNet", seq_len=8, bg_mode="subtract")
        x = torch.randn(1, 8, HEIGHT, WIDTH)
        out = model(x)
        assert out.shape == (1, 8, HEIGHT, WIDTH)

    def test_tracknet_subtract_concat(self):
        model = get_model("TrackNet", seq_len=8, bg_mode="subtract_concat")
        x = torch.randn(1, 32, HEIGHT, WIDTH)
        out = model(x)
        assert out.shape == (1, 8, HEIGHT, WIDTH)

    def test_tracknet_concat(self):
        model = get_model("TrackNet", seq_len=8, bg_mode="concat")
        x = torch.randn(1, 27, HEIGHT, WIDTH)
        out = model(x)
        assert out.shape == (1, 8, HEIGHT, WIDTH)

    def test_inpaintnet(self):
        model = get_model("InpaintNet")
        assert isinstance(model, InpaintNet)

    def test_invalid_model_name(self):
        with pytest.raises(ValueError, match="Invalid model name"):
            get_model("FakeModel")


class TestToImg:
    def test_converts_0_1_to_0_255(self):
        img = np.array([0.0, 0.5, 1.0])
        result = to_img(img)
        np.testing.assert_array_equal(result, np.array([0, 127, 255]))
        assert result.dtype == np.uint8

    def test_preserves_shape(self):
        img = np.random.rand(HEIGHT, WIDTH, 3)
        result = to_img(img)
        assert result.shape == (HEIGHT, WIDTH, 3)


class TestToImgFormat:
    def test_single_channel(self):
        """num_ch=1: output should be identical to input (N, L, H, W)."""
        x = np.random.rand(2, 8, HEIGHT, WIDTH)
        result = to_img_format(x, num_ch=1)
        np.testing.assert_array_equal(result, x)

    def test_three_channel(self):
        """num_ch=3: (N, L*3, H, W) -> (N, L, H, W, 3)."""
        seq_len = 4
        x = np.random.rand(2, seq_len * 3, HEIGHT, WIDTH)
        result = to_img_format(x, num_ch=3)
        assert result.shape == (2, seq_len, HEIGHT, WIDTH, 3)

    def test_rejects_non_4d(self):
        with pytest.raises(AssertionError):
            to_img_format(np.random.rand(8, HEIGHT, WIDTH), num_ch=1)


class TestListDirs:
    def test_returns_sorted_full_paths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create some subdirs
            for name in ["c_dir", "a_dir", "b_dir"]:
                os.makedirs(os.path.join(tmpdir, name))
            # Create a file too
            with open(os.path.join(tmpdir, "file.txt"), "w") as f:
                f.write("test")

            result = list_dirs(tmpdir)
            assert len(result) == 4
            assert result == sorted(result)
            assert all(os.path.isabs(p) for p in result)

    def test_empty_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = list_dirs(tmpdir)
            assert result == []


class TestResumeArgumentParser:
    def test_parses_all_fields(self):
        param_dict = {
            "model_name": "TrackNet",
            "seq_len": 8,
            "epochs": 50,
            "batch_size": 10,
            "optim": "Adam",
            "learning_rate": 0.001,
            "lr_scheduler": "StepLR",
            "bg_mode": "concat",
            "alpha": 0.5,
            "frame_alpha": -1,
            "mask_ratio": 0.3,
            "tolerance": 4.0,
            "resume_training": "",
            "seed": 42,
            "save_dir": "experiments/test",
            "debug": False,
            "verbose": True,
        }
        parser = ResumeArgumentParser(param_dict)
        assert parser.model_name == "TrackNet"
        assert parser.seq_len == 8
        assert parser.epochs == 50
        assert parser.bg_mode == "concat"
        assert parser.seed == 42


class TestConstants:
    def test_height_width(self):
        assert HEIGHT == 288
        assert WIDTH == 512

    def test_sigma(self):
        assert SIGMA == 2.5
