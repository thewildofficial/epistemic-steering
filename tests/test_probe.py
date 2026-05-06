import json
import numpy as np
import pytest
import tempfile
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from probe import (
    load_probe_weights,
    compute_confidence,
    compute_auroc,
    compute_confusion_matrix,
    compute_prevention_rate,
    compute_unnecessary_block_rate,
    threshold_sweep,
)


class TestLoadProbeWeights:
    def test_load_json_weights(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            data = {
                'coef': [1.0] * 2560,
                'intercept': 0.5,
                'layer': 30,
                'token_position': 'last'
            }
            json.dump(data, f)
            temp_path = f.name

        weights = load_probe_weights(temp_path)
        assert weights['coef'].shape == (2560,)
        assert weights['intercept'] == 0.5
        assert weights['layer'] == 30
        assert weights['token_position'] == 'last'
        Path(temp_path).unlink()

    def test_load_npz_weights(self):
        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            np.savez(f, coef=np.ones(2560), intercept=0.3, layer=25, token_position='allpos')
            temp_path = f.name

        weights = load_probe_weights(temp_path)
        assert weights['coef'].shape == (2560,)
        assert weights['intercept'] == 0.3
        assert weights['layer'] == 25
        assert weights['token_position'] == 'allpos'
        Path(temp_path).unlink()


class TestComputeConfidence:
    def test_output_in_range(self):
        weights = {'coef': np.zeros(2560), 'intercept': 0.0}
        activation = np.zeros(2560)
        conf = compute_confidence(activation, weights)
        assert 0.0 <= conf <= 1.0

    def test_high_confidence(self):
        weights = {'coef': np.zeros(2560), 'intercept': 10.0}
        activation = np.zeros(2560)
        conf = compute_confidence(activation, weights)
        assert conf > 0.99

    def test_low_confidence(self):
        weights = {'coef': np.zeros(2560), 'intercept': -10.0}
        activation = np.zeros(2560)
        conf = compute_confidence(activation, weights)
        assert conf < 0.01

    def test_sigmoid_properties(self):
        weights = {'coef': np.zeros(2560), 'intercept': 0.0}
        activation = np.zeros(2560)
        conf = compute_confidence(activation, weights)
        assert abs(conf - 0.5) < 1e-6


class TestComputeAuroc:
    def test_perfect_scores(self):
        scores = np.array([0.9, 0.8, 0.7, 0.6])
        labels = np.array([True, True, False, False])
        auroc = compute_auroc(scores, labels)
        assert auroc == 1.0

    def test_perfect_inverse(self):
        scores = np.array([0.9, 0.8, 0.7, 0.6])
        labels = np.array([False, False, True, True])
        auroc = compute_auroc(scores, labels)
        assert auroc == 0.0

    def test_random_scores(self):
        np.random.seed(42)
        scores = np.random.rand(100)
        labels = np.random.rand(100) > 0.5
        auroc = compute_auroc(scores, labels)
        assert 0.0 <= auroc <= 1.0

    def test_all_same_label(self):
        scores = np.array([0.1, 0.5, 0.9])
        labels = np.array([True, True, True])
        auroc = compute_auroc(scores, labels)
        assert np.isnan(auroc)


class TestComputeConfusionMatrix:
    def test_perfect_confident(self):
        scores = np.array([0.9, 0.8, 0.1, 0.2])
        labels = np.array([True, True, False, False])
        cm = compute_confusion_matrix(scores, labels, threshold=0.5)
        assert cm['TP'] == 2
        assert cm['TN'] == 2
        assert cm['FP'] == 0
        assert cm['FN'] == 0

    def test_all_wrong(self):
        scores = np.array([0.9, 0.8, 0.1, 0.2])
        labels = np.array([False, False, True, True])
        cm = compute_confusion_matrix(scores, labels, threshold=0.5)
        assert cm['FP'] == 2
        assert cm['FN'] == 2

    def test_threshold_shift(self):
        scores = np.array([0.5, 0.6, 0.4])
        labels = np.array([True, False, True])
        cm = compute_confusion_matrix(scores, labels, threshold=0.55)
        assert cm['TP'] == 0
        assert cm['FP'] == 1
        assert cm['TN'] == 0
        assert cm['FN'] == 2


class TestPreventionRate:
    def test_all_caught(self):
        scores = np.array([0.1, 0.2, 0.3])
        labels = np.array([False, False, False])
        rate = compute_prevention_rate(scores, labels, threshold=0.5)
        assert rate == 1.0

    def test_none_caught(self):
        scores = np.array([0.9, 0.8, 0.7])
        labels = np.array([False, False, False])
        rate = compute_prevention_rate(scores, labels, threshold=0.5)
        assert rate == 0.0

    def test_partial(self):
        scores = np.array([0.1, 0.8, 0.3, 0.9])
        labels = np.array([False, False, False, False])
        rate = compute_prevention_rate(scores, labels, threshold=0.5)
        assert rate == 0.5

    def test_no_incorrect(self):
        scores = np.array([0.1, 0.5, 0.9])
        labels = np.array([True, True, True])
        rate = compute_prevention_rate(scores, labels, threshold=0.5)
        assert np.isnan(rate)


class TestUnnecessaryBlockRate:
    def test_all_blocked(self):
        scores = np.array([0.1, 0.2, 0.3])
        labels = np.array([True, True, True])
        rate = compute_unnecessary_block_rate(scores, labels, threshold=0.5)
        assert rate == 1.0

    def test_none_blocked(self):
        scores = np.array([0.9, 0.8, 0.7])
        labels = np.array([True, True, True])
        rate = compute_unnecessary_block_rate(scores, labels, threshold=0.5)
        assert rate == 0.0

    def test_partial(self):
        scores = np.array([0.1, 0.8, 0.3, 0.9])
        labels = np.array([True, True, True, True])
        rate = compute_unnecessary_block_rate(scores, labels, threshold=0.5)
        assert rate == 0.5

    def test_no_correct(self):
        scores = np.array([0.1, 0.5, 0.9])
        labels = np.array([False, False, False])
        rate = compute_unnecessary_block_rate(scores, labels, threshold=0.5)
        assert np.isnan(rate)


class TestThresholdSweep:
    def test_default_thresholds(self):
        scores = np.array([0.1, 0.3, 0.5, 0.7, 0.9] * 20)
        labels = np.array([False, False, False, True, True] * 20)
        df = threshold_sweep(scores, labels)
        assert len(df) > 0

    def test_columns_present(self):
        scores = np.random.rand(100)
        labels = np.random.rand(100) > 0.5
        df = threshold_sweep(scores, labels)
        expected = {'threshold', 'auroc', 'precision', 'recall', 'f1', 'prevention_rate', 'unnecessary_block_rate'}
        assert expected.issubset(set(df.columns))

    def test_threshold_monotonic(self):
        scores = np.array([0.1, 0.3, 0.5, 0.7, 0.9] * 20)
        labels = np.array([False, False, False, True, True] * 20)
        df = threshold_sweep(scores, labels)
        thresholds = df['threshold'].values
        assert np.all(thresholds[:-1] < thresholds[1:])


class TestEdgeCases:
    def test_empty_scores(self):
        scores = np.array([])
        labels = np.array([])
        with pytest.raises(Exception):
            compute_confusion_matrix(scores, labels)

    def test_threshold_zero(self):
        scores = np.array([0.0, 0.5, 1.0])
        labels = np.array([True, False, True])
        cm = compute_confusion_matrix(scores, labels, threshold=0.0)
        assert cm['TP'] == 2
        assert cm['FP'] == 1
        assert cm['TN'] == 0
        assert cm['FN'] == 0

    def test_threshold_one(self):
        scores = np.array([0.0, 0.5, 1.0])
        labels = np.array([True, False, True])
        cm = compute_confusion_matrix(scores, labels, threshold=1.0)
        assert cm['TP'] == 1
        assert cm['FP'] == 0
        assert cm['TN'] == 1
        assert cm['FN'] == 1