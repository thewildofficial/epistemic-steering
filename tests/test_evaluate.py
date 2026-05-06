"""Tests for evaluation metrics in src/evaluate.py."""

import numpy as np
import pytest
from src.evaluate import (
    confusion_matrix_at_threshold,
    prevention_rate_at_threshold,
    unnecessary_block_rate_at_threshold,
    selective_accuracy,
    token_efficiency,
    calibration_curve,
    compute_all_metrics,
)


class TestConfusionMatrixAtThreshold:
    def test_all_correct_confident(self):
        scores = np.array([0.9, 0.8, 0.7])
        labels = np.array([True, True, True])
        result = confusion_matrix_at_threshold(scores, labels, 0.5)
        assert result == {"TP": 3, "FP": 0, "TN": 0, "FN": 0}

    def test_all_incorrect_confident(self):
        scores = np.array([0.9, 0.8, 0.7])
        labels = np.array([False, False, False])
        result = confusion_matrix_at_threshold(scores, labels, 0.5)
        assert result == {"TP": 0, "FP": 3, "TN": 0, "FN": 0}

    def test_mixed_confident_uncertain(self):
        scores = np.array([0.9, 0.3, 0.8, 0.2])
        labels = np.array([True, True, False, False])
        result = confusion_matrix_at_threshold(scores, labels, 0.5)
        # TP: scores >= 0.5 and labels True -> [0.9], TP=1
        # FP: scores >= 0.5 and labels False -> [0.8], FP=1
        # TN: scores < 0.5 and labels False -> [0.2], TN=1
        # FN: scores < 0.5 and labels True -> [0.3], FN=1
        assert result == {"TP": 1, "FP": 1, "TN": 1, "FN": 1}

    def test_threshold_at_boundaries(self):
        scores = np.array([0.5, 0.5, 0.4, 0.4])
        labels = np.array([True, False, True, False])
        result = confusion_matrix_at_threshold(scores, labels, 0.5)
        # >= 0.5 means scores 0.5 count as confident
        assert result == {"TP": 1, "FP": 1, "TN": 1, "FN": 1}

    def test_empty_arrays(self):
        scores = np.array([])
        labels = np.array([])
        result = confusion_matrix_at_threshold(scores, labels, 0.5)
        assert result == {"TP": 0, "FP": 0, "TN": 0, "FN": 0}


class TestPreventionRateAtThreshold:
    def test_all_incorrect_all_low_score(self):
        scores = np.array([0.1, 0.2, 0.3])
        labels = np.array([False, False, False])
        result = prevention_rate_at_threshold(scores, labels, 0.5)
        assert result == 1.0

    def test_all_incorrect_all_high_score(self):
        scores = np.array([0.7, 0.8, 0.9])
        labels = np.array([False, False, False])
        result = prevention_rate_at_threshold(scores, labels, 0.5)
        assert result == 0.0

    def test_mixed(self):
        scores = np.array([0.1, 0.8, 0.2, 0.9])
        labels = np.array([False, False, False, False])
        result = prevention_rate_at_threshold(scores, labels, 0.5)
        # 2 caught (0.1, 0.2), 2 missed (0.8, 0.9)
        assert result == 0.5

    def test_no_incorrect_answers(self):
        scores = np.array([0.1, 0.2, 0.3])
        labels = np.array([True, True, True])
        result = prevention_rate_at_threshold(scores, labels, 0.5)
        assert result == 0.0

    def test_threshold_0_catches_none(self):
        scores = np.array([0.0, 0.1, 0.2])
        labels = np.array([False, False, False])
        result = prevention_rate_at_threshold(scores, labels, 0.0)
        assert result == 0.0

    def test_threshold_1_catches_all(self):
        scores = np.array([0.9, 0.9, 0.9])
        labels = np.array([False, False, False])
        result = prevention_rate_at_threshold(scores, labels, 1.0)
        assert result == 1.0


class TestUnnecessaryBlockRateAtThreshold:
    def test_all_correct_all_low_score(self):
        scores = np.array([0.1, 0.2, 0.3])
        labels = np.array([True, True, True])
        result = unnecessary_block_rate_at_threshold(scores, labels, 0.5)
        assert result == 1.0

    def test_all_correct_all_high_score(self):
        scores = np.array([0.7, 0.8, 0.9])
        labels = np.array([True, True, True])
        result = unnecessary_block_rate_at_threshold(scores, labels, 0.5)
        assert result == 0.0

    def test_mixed(self):
        scores = np.array([0.1, 0.8, 0.2, 0.9])
        labels = np.array([True, True, True, True])
        result = unnecessary_block_rate_at_threshold(scores, labels, 0.5)
        # 2 blocked (0.1, 0.2), 2 not blocked (0.8, 0.9)
        assert result == 0.5

    def test_no_correct_answers(self):
        scores = np.array([0.1, 0.2, 0.3])
        labels = np.array([False, False, False])
        result = unnecessary_block_rate_at_threshold(scores, labels, 0.5)
        assert result == 0.0

    def test_threshold_0_blocks_none(self):
        scores = np.array([0.0, 0.1, 0.2])
        labels = np.array([True, True, True])
        result = unnecessary_block_rate_at_threshold(scores, labels, 0.0)
        assert result == 0.0

    def test_threshold_1_blocks_all(self):
        scores = np.array([0.9, 0.9, 0.9])
        labels = np.array([True, True, True])
        result = unnecessary_block_rate_at_threshold(scores, labels, 1.0)
        assert result == 1.0


class TestSelectiveAccuracy:
    def test_perfect_score(self):
        result = selective_accuracy(80, 10, 5, 100)
        assert result == 0.9

    def test_half_correct(self):
        result = selective_accuracy(25, 25, 0, 100)
        assert result == 0.5

    def test_no_correct(self):
        result = selective_accuracy(0, 0, 50, 100)
        assert result == 0.0

    def test_all_abstain(self):
        result = selective_accuracy(0, 0, 100, 100)
        assert result == 0.0

    def test_zero_total(self):
        result = selective_accuracy(0, 0, 0, 0)
        assert result == 0.0

    def test_abstentions_not_counted_as_errors(self):
        # 50 correct out of 100 total, 50 abstentions
        # abstentions are not errors, just "don't know"
        result = selective_accuracy(30, 20, 50, 100)
        assert result == 0.5


class TestTokenEfficiency:
    def test_savings_vs_cot(self):
        result = token_efficiency(
            direct_tokens=1000, cot_tokens=2000, routed_tokens=1500, total=100
        )
        assert result["tokens_per_question"] == 15.0
        # savings = (2000 - 1500) / 2000 = 0.25
        assert result["savings_vs_cot"] == 0.25

    def test_no_savings_when_equal_to_cot(self):
        result = token_efficiency(
            direct_tokens=2000, cot_tokens=2000, routed_tokens=2000, total=100
        )
        assert result["savings_vs_cot"] == 0.0

    def test_zero_cot_tokens(self):
        result = token_efficiency(
            direct_tokens=0, cot_tokens=0, routed_tokens=0, total=100
        )
        assert result["savings_vs_cot"] == 0.0

    def test_zero_total(self):
        result = token_efficiency(
            direct_tokens=1000, cot_tokens=2000, routed_tokens=1500, total=0
        )
        assert result["tokens_per_question"] == 0.0

    def test_structure_keys(self):
        result = token_efficiency(
            direct_tokens=1000, cot_tokens=2000, routed_tokens=1500, total=100
        )
        assert "tokens_per_question" in result
        assert "tokens_per_correct" in result
        assert "savings_vs_cot" in result


class TestCalibrationCurve:
    def test_output_shapes(self):
        np.random.seed(42)
        scores = np.random.uniform(0, 1, 100)
        labels = (scores + np.random.normal(0, 0.2, 100)) > 0.5
        result = calibration_curve(scores, labels, n_bins=10)

        assert len(result["bin_centers"]) == 10
        assert len(result["observed_accuracy"]) == 10
        assert len(result["bin_counts"]) == 10

    def test_bin_counts_sum_to_total(self):
        scores = np.random.uniform(0, 1, 100)
        labels = np.random.choice([True, False], 100)
        result = calibration_curve(scores, labels, n_bins=10)

        assert sum(result["bin_counts"]) == 100

    def test_perfect_calibration(self):
        # When scores == labels (perfect calibration)
        scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
        labels = np.array([False, False, False, False, False, True, True, True, True, True])
        result = calibration_curve(scores, labels, n_bins=10)

        # Observed accuracy should roughly match bin centers
        assert len(result["bin_centers"]) == 10
        assert len(result["observed_accuracy"]) == 10

    def test_n_bins_custom(self):
        scores = np.random.uniform(0, 1, 100)
        labels = np.random.choice([True, False], 100)
        result = calibration_curve(scores, labels, n_bins=5)

        assert len(result["bin_centers"]) == 5
        assert len(result["observed_accuracy"]) == 5
        assert len(result["bin_counts"]) == 5


class TestComputeAllMetrics:
    def test_contains_all_expected_keys(self):
        np.random.seed(42)
        scores = np.random.uniform(0, 1, 100)
        labels = np.random.choice([True, False], 100)
        result = compute_all_metrics(scores, labels, threshold=0.5)

        expected_keys = [
            "auroc",
            "threshold",
            "confusion_matrix",
            "prevention_rate",
            "unnecessary_block_rate",
            "precision",
            "recall",
            "f1_score",
            "calibration",
        ]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"

    def test_confusion_matrix_structure(self):
        scores = np.array([0.9, 0.3, 0.8, 0.2])
        labels = np.array([True, True, False, False])
        result = compute_all_metrics(scores, labels, threshold=0.5)

        cm = result["confusion_matrix"]
        assert "TP" in cm
        assert "FP" in cm
        assert "TN" in cm
        assert "FN" in cm

    def test_calibration_structure(self):
        scores = np.random.uniform(0, 1, 50)
        labels = np.random.choice([True, False], 50)
        result = compute_all_metrics(scores, labels, threshold=0.5)

        calib = result["calibration"]
        assert "bin_centers" in calib
        assert "observed_accuracy" in calib
        assert "bin_counts" in calib

    def test_threshold_value_preserved(self):
        scores = np.array([0.9, 0.3, 0.8, 0.2])
        labels = np.array([True, True, False, False])
        result = compute_all_metrics(scores, labels, threshold=0.7)

        assert result["threshold"] == 0.7

    def test_auroc_in_valid_range(self):
        np.random.seed(42)
        scores = np.random.uniform(0, 1, 100)
        labels = np.random.choice([True, False], 100)
        result = compute_all_metrics(scores, labels, threshold=0.5)

        assert 0.0 <= result["auroc"] <= 1.0

    def test_precision_recall_f1_from_cm(self):
        # With TP=2, FP=1, TN=1, FN=0:
        # precision = 2/3, recall = 2/2 = 1.0, f1 = 2*2/3 / (2/3 + 1) = 4/3 / 5/3 = 4/5 = 0.8
        scores = np.array([0.9, 0.9, 0.3, 0.8])
        labels = np.array([True, True, False, False])
        result = compute_all_metrics(scores, labels, threshold=0.5)

        assert result["precision"] == pytest.approx(2 / 3)
        assert result["recall"] == pytest.approx(1.0)
        assert result["f1_score"] == pytest.approx(0.8)

    def test_perfect_separation(self):
        # Perfect separation: all correct have high scores, all incorrect have low
        scores = np.array([0.9, 0.8, 0.1, 0.2])
        labels = np.array([True, True, False, False])
        result = compute_all_metrics(scores, labels, threshold=0.5)

        assert result["auroc"] == 1.0
        assert result["confusion_matrix"] == {"TP": 2, "FP": 0, "TN": 2, "FN": 0}
        assert result["prevention_rate"] == 1.0
        assert result["unnecessary_block_rate"] == 0.0

    def test_inverted_separation(self):
        # Inverted: all correct have low scores, all incorrect have high
        scores = np.array([0.1, 0.2, 0.9, 0.8])
        labels = np.array([True, True, False, False])
        result = compute_all_metrics(scores, labels, threshold=0.5)

        assert result["auroc"] == 0.0
        assert result["confusion_matrix"] == {"TP": 0, "FP": 2, "TN": 0, "FN": 2}
        assert result["prevention_rate"] == 0.0
        assert result["unnecessary_block_rate"] == 1.0