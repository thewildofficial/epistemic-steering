"""Tests for plotting module.

Tests the 6 plotting functions in src/plotting.py to ensure they
work correctly and produce expected output files.
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.plotting import (
    plot_auroc_curve,
    plot_confusion_matrix_heatmap,
    plot_threshold_tradeoff,
    plot_calibration_curve,
    plot_dataset_comparison,
    plot_prevention_rate_curve
)


@pytest.fixture
def temp_dir():
    """Create temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


class TestPlotAurocCurve:
    """Test ROC curve plotting."""
    
    def test_basic_plot(self, temp_dir):
        """Test basic ROC curve generation."""
        fpr = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        tpr = np.array([0, 0.3, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0])
        
        plot_auroc_curve(fpr, tpr, 0.85, "Test ROC", str(temp_dir / "test_roc"))
        
        assert (temp_dir / "test_roc.png").exists()
        assert (temp_dir / "test_roc.pdf").exists()
    
    def test_perfect_curve(self, temp_dir):
        """Test perfect ROC curve (AUROC=1.0)."""
        fpr = np.array([0, 0, 1])
        tpr = np.array([0, 1, 1])
        
        plot_auroc_curve(fpr, tpr, 1.0, "Perfect ROC", str(temp_dir / "perfect_roc"))
        
        assert (temp_dir / "perfect_roc.png").exists()


class TestPlotConfusionMatrixHeatmap:
    """Test confusion matrix heatmap plotting."""
    
    def test_basic_heatmap(self, temp_dir):
        """Test basic confusion matrix generation."""
        cm = {'TP': 100, 'FP': 20, 'TN': 80, 'FN': 10}
        labels = ['Negative', 'Positive']
        
        plot_confusion_matrix_heatmap(cm, labels, "Test CM", str(temp_dir / "test_cm"))
        
        assert (temp_dir / "test_cm.png").exists()
        assert (temp_dir / "test_cm.pdf").exists()


class TestPlotThresholdTradeoff:
    """Test threshold tradeoff plotting."""
    
    def test_basic_tradeoff(self, temp_dir):
        """Test basic threshold tradeoff generation."""
        thresholds = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        prevention = np.array([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95])
        unnecessary_block = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45])
        
        plot_threshold_tradeoff(
            thresholds, prevention, unnecessary_block, 0.5,
            str(temp_dir / "test_tradeoff")
        )
        
        assert (temp_dir / "test_tradeoff.png").exists()
        assert (temp_dir / "test_tradeoff.pdf").exists()


class TestPlotCalibrationCurve:
    """Test calibration curve plotting."""
    
    def test_basic_calibration(self, temp_dir):
        """Test basic calibration curve generation."""
        bin_centers = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        observed_accuracy = np.array([0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95])
        
        plot_calibration_curve(
            bin_centers, observed_accuracy, "Test Calibration",
            str(temp_dir / "test_calibration")
        )
        
        assert (temp_dir / "test_calibration.png").exists()
        assert (temp_dir / "test_calibration.pdf").exists()
    
    def test_with_bin_counts(self, temp_dir):
        """Test calibration curve with bin counts."""
        bin_centers = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        observed_accuracy = np.array([0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95])
        bin_counts = np.array([100, 80, 60, 50, 40, 30, 20, 10, 5])
        
        plot_calibration_curve(
            bin_centers, observed_accuracy, "Test Calibration with Counts",
            str(temp_dir / "test_calibration_counts"),
            bin_counts=bin_counts
        )
        
        assert (temp_dir / "test_calibration_counts.png").exists()


class TestPlotDatasetComparison:
    """Test dataset comparison plotting."""
    
    def test_basic_comparison(self, temp_dir):
        """Test basic dataset comparison generation."""
        mmlu_metrics = {
            'auroc': 0.85,
            'prevention_rate': 0.75,
            'best_threshold': 0.5
        }
        gsm8k_metrics = {
            'auroc': 0.95,
            'prevention_rate': 0.90,
            'best_threshold': 0.6
        }
        
        plot_dataset_comparison(
            mmlu_metrics, gsm8k_metrics,
            str(temp_dir / "test_comparison")
        )
        
        assert (temp_dir / "test_comparison.png").exists()
        assert (temp_dir / "test_comparison.pdf").exists()


class TestPlotPreventionRateCurve:
    """Test prevention rate curve plotting."""
    
    def test_basic_prevention(self, temp_dir):
        """Test basic prevention rate curve generation."""
        thresholds = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        prevention_rates = np.array([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95])
        
        plot_prevention_rate_curve(
            thresholds, prevention_rates, 0.5,
            str(temp_dir / "test_prevention"),
            dataset_name="Test"
        )
        
        assert (temp_dir / "test_prevention.png").exists()
        assert (temp_dir / "test_prevention.pdf").exists()
    
    def test_without_dataset_name(self, temp_dir):
        """Test prevention rate curve without dataset name."""
        thresholds = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        prevention_rates = np.array([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95])
        
        plot_prevention_rate_curve(
            thresholds, prevention_rates, 0.5,
            str(temp_dir / "test_prevention_no_name")
        )
        
        assert (temp_dir / "test_prevention_no_name.png").exists()