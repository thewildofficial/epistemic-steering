"""Tests for mean_pool — mean pooling over hidden state sequences."""

import numpy as np
import pytest
from src.mean_pool import mean_pool


class TestMeanPool:
    def test_basic_mean_pool_shape(self):
        arr = np.random.randn(50, 2560)
        result = mean_pool(arr)
        assert result.shape == (2560,)

    def test_correct_mean_values(self):
        arr = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=float)
        result = mean_pool(arr)
        expected = np.array([3.0, 4.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_single_token(self):
        arr = np.array([[7.0, 8.0, 9.0]], dtype=float)
        result = mean_pool(arr)
        np.testing.assert_array_almost_equal(result, arr[0])

    def test_empty_sequence_returns_zero_vector(self):
        arr = np.zeros((0, 2560), dtype=float)
        result = mean_pool(arr)
        assert result.shape == (2560,)
        np.testing.assert_array_equal(result, np.zeros(2560))

    def test_known_values(self):
        arr = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=float)
        result = mean_pool(arr)
        np.testing.assert_array_almost_equal(result, [20.0, 30.0])

    def test_raises_on_1d_input(self):
        with pytest.raises(ValueError):
            mean_pool(np.array([1.0, 2.0, 3.0]))
