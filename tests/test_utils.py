"""Tests for utility functions in src/utils/general.py."""

import os
import tempfile

import numpy as np

from src.utils.general import list_dirs, to_img


class TestToImg:
    def test_converts_0_1_to_0_255(self):
        img = np.array([0.0, 0.5, 1.0])
        result = to_img(img)
        np.testing.assert_array_equal(result, np.array([0, 127, 255]))
        assert result.dtype == np.uint8

    def test_preserves_shape(self):
        img = np.random.rand(288, 512, 3)
        result = to_img(img)
        assert result.shape == (288, 512, 3)


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
