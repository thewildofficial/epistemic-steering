"""Tests for hybrid_backoff — z-score normalization, alpha computation,
and hybrid score combination."""

import pytest
from src.hybrid_backoff import zscore_normalize, compute_alpha, hybrid_score


class TestZscoreNormalize:
    def test_basic_normalization(self):
        values = [1.0, 2.0, 3.0]
        result = zscore_normalize(values)
        assert len(result) == 3
        # mean=2, std≈0.816 → [-1.22, 0, 1.22]
        assert result[0] < result[1] < result[2]
        assert abs(result[1]) < 1e-6  # middle value at mean

    def test_constant_values_return_zeros(self):
        result = zscore_normalize([5.0, 5.0, 5.0])
        assert result == [0.0, 0.0, 0.0]

    def test_single_value(self):
        result = zscore_normalize([42.0])
        assert result == [0.0]

    def test_empty_list(self):
        result = zscore_normalize([])
        assert result == []

    def test_negative_values(self):
        values = [-5.0, 0.0, 5.0]
        result = zscore_normalize(values)
        assert result[0] < 0.0
        assert abs(result[1]) < 1e-6
        assert result[2] > 0.0


class TestComputeAlpha:
    def test_in_domain_alpha_is_0_7(self):
        alpha = compute_alpha(0.8, ood_alarm=False)
        assert alpha == pytest.approx(0.7)

    def test_ood_alarm_alpha_drops_to_zero(self):
        alpha = compute_alpha(0.8, ood_alarm=True)
        assert alpha == pytest.approx(0.0)

    def test_low_probe_still_0_7_in_domain(self):
        alpha = compute_alpha(0.2, ood_alarm=False)
        assert alpha == pytest.approx(0.7)


class TestHybridScore:
    def test_combination_formula_structure(self):
        # With alpha=1.0, only probe matters
        score = hybrid_score(probe_score=1.0, msp=0.9, entropy=0.5, alpha=1.0)
        result = zscore_normalize([1.0, 0.9, 0.5])
        assert score == pytest.approx(result[0])

    def test_alpha_zero_gives_msp_entropy_mix(self):
        score = hybrid_score(probe_score=1.0, msp=0.9, entropy=0.5, alpha=0.0)
        result = zscore_normalize([1.0, 0.9, 0.5])
        expected = 0.5 * result[1] + 0.5 * result[2]
        assert score == pytest.approx(expected)

    def test_alpha_0_7_blend(self):
        score = hybrid_score(probe_score=0.8, msp=0.7, entropy=0.3, alpha=0.7)
        result = zscore_normalize([0.8, 0.7, 0.3])
        expected = 0.7 * result[0] + 0.3 * (0.5 * result[1] + 0.5 * result[2])
        assert score == pytest.approx(expected)

    def test_scores_are_floats(self):
        score = hybrid_score(probe_score=0.5, msp=0.5, entropy=0.5, alpha=0.5)
        assert isinstance(score, float)
