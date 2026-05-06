"""Probe scoring and uncertainty quantification.

Computes epistemic confidence scores from LLM hidden states using
logistic regression probes trained on the Qwen3.5-4B model.

Core functions:
- load_probe_weights: Load trained probe coefficients
- compute_confidence: g(h_ℓ(x)) = σ(w^T h_ℓ(x) + b) ∈ [0,1]
- score_dataset: Batch confidence scoring
- threshold_sweep: Metric computation across thresholds
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.special import expit
from sklearn.metrics import roc_auc_score

# Hidden dimension for Qwen3.5-4B layer 30
HIDDEN_DIM = 2560


def load_probe_weights(path: str) -> dict:
    """Load trained probe weights from JSON/npz file.

    Args:
        path: Path to probe weights file (.json or .npz)

    Returns:
        dict with keys:
            - coef: np.ndarray of shape (2560,)
            - intercept: float
            - layer: int
            - token_position: str
    """
    path = Path(path)
    if path.suffix == '.npz':
        data = np.load(path)
        return {
            'coef': data['coef'],
            'intercept': float(data['intercept']),
            'layer': int(data.get('layer', 30)),
            'token_position': str(data.get('token_position', 'last'))
        }
    else:
        with open(path, 'r') as f:
            data = json.load(f)
        return {
            'coef': np.array(data['coef']),
            'intercept': float(data['intercept']),
            'layer': int(data.get('layer', 30)),
            'token_position': str(data.get('token_position', 'last'))
        }


def compute_confidence(activation: np.ndarray, weights: dict) -> float:
    """Compute epistemic confidence score.

    Computes g(h_ℓ(x)) = σ(w^T h_ℓ(x) + b) ∈ [0,1]

    Args:
        activation: Hidden state vector of shape (2560,)
        weights: Dict with 'coef' (2560,) and 'intercept' (float)

    Returns:
        float confidence score in [0, 1]
    """
    coef = weights['coef']
    intercept = weights['intercept']

    # Compute linear combination: w^T h + b
    score = np.dot(coef, activation) + intercept

    # Apply sigmoid with numerical stability via scipy.special.expit
    return float(expit(score))


def score_dataset(
    activations_dir: str,
    weights: dict,
    layer: int = 30,
    token_position: str = 'last'
) -> np.ndarray:
    """Score all questions in an activations directory.

    Args:
        activations_dir: Directory containing .npy activation files
        weights: Probe weights dict from load_probe_weights
        layer: Layer number (used for filename pattern)
        token_position: Token position ('last' or 'allpos')

    Returns:
        np.ndarray of confidence scores, one per question
    """
    activations_dir = Path(activations_dir)
    scores = []

    # Pattern: q{question_id}_layer{layer}.npy
    pattern = f"*_layer{layer}.npy"
    for npy_path in sorted(activations_dir.glob(pattern)):
        activation = np.load(npy_path)
        # If activation is 2D (sequence, hidden), take last token
        if activation.ndim == 2:
            activation = activation[-1]
        score = compute_confidence(activation, weights)
        scores.append(score)

    return np.array(scores)


def compute_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Compute Area Under ROC Curve.

    Args:
        scores: Confidence scores from probe
        labels: Ground truth labels (True = correct, False = hallucination)

    Returns:
        float AUROC score, or float('nan') if all labels are same
    """
    # Handle edge case: all same label
    if len(np.unique(labels)) == 1:
        return float('nan')

    return float(roc_auc_score(labels, scores))


def compute_confusion_matrix(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.5
) -> dict:
    """Compute confusion matrix components.

    "Positive" = model was correct (label is True)
    "Confident" = probe score >= threshold

    Args:
        scores: Confidence scores from probe
        labels: Ground truth labels (True = correct, False = hallucination)
        threshold: Decision threshold for probe confidence

    Returns:
        dict with TP, FP, TN, FN counts
    """
    confident = scores >= threshold

    # True Positive: correct (label=True) and confident (probe agrees)
    TP = int(np.sum(confident & labels))

    # False Positive: incorrect (label=False) but confident (probe wrong)
    FP = int(np.sum(confident & ~labels))

    # True Negative: incorrect (label=False) and not confident (probe caught it)
    TN = int(np.sum(~confident & ~labels))

    # False Negative: correct (label=True) but not confident (probe missed it)
    FN = int(np.sum(~confident & labels))

    return {'TP': TP, 'FP': FP, 'TN': TN, 'FN': FN}


def compute_prevention_rate(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.5
) -> float:
    """Compute hallucination prevention rate.

    Fraction of actual hallucinations (label=False) that would be caught
    by the probe (low confidence = score < threshold).

    Among questions where the model was WRONG (label=False),
    what fraction had low probe score (< threshold)?

    Prevention rate = (label=False AND score < threshold) / (label=False)
                    = TN / (TN + FP)

    Args:
        scores: Confidence scores from probe
        labels: Ground truth labels (True = correct, False = hallucination)
        threshold: Decision threshold

    Returns:
        float fraction in [0, 1]
    """
    incorrect_mask = ~labels
    incorrect_count = np.sum(incorrect_mask)

    if incorrect_count == 0:
        return float('nan')

    # Caught = low confidence when incorrect = TN
    caught = np.sum(incorrect_mask & (scores < threshold))
    return float(caught / incorrect_count)


def compute_unnecessary_block_rate(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.5
) -> float:
    """Compute unnecessary block rate.

    Fraction of correct answers (label=True) that would be unnecessarily
    blocked by the probe (low confidence despite correct answer).

    Among questions where the model was CORRECT (label=True),
    what fraction had low probe score (< threshold)?

    Block rate = (label=True AND score < threshold) / (label=True)
               = FN / (FN + TP)

    Args:
        scores: Confidence scores from probe
        labels: Ground truth labels (True = correct, False = hallucination)
        threshold: Decision threshold

    Returns:
        float fraction in [0, 1]
    """
    correct_mask = labels
    correct_count = np.sum(correct_mask)

    if correct_count == 0:
        return float('nan')

    # Unnecessarily blocked = correct but low confidence = FN
    blocked = np.sum(correct_mask & (scores < threshold))
    return float(blocked / correct_count)


def threshold_sweep(
    scores: np.ndarray,
    labels: np.ndarray,
    thresholds: np.ndarray | None = None
) -> pd.DataFrame:
    """Compute metrics across a range of thresholds.

    Args:
        scores: Confidence scores from probe
        labels: Ground truth labels
        thresholds: Array of threshold values, or None for default (0.1 to 0.95 step 0.05)

    Returns:
        pd.DataFrame with columns:
            threshold, auroc, precision, recall, f1, prevention_rate, unnecessary_block_rate
    """
    if thresholds is None:
        thresholds = np.arange(0.1, 1.0, 0.05)

    results = []
    for thresh in thresholds:
        cm = compute_confusion_matrix(scores, labels, thresh)
        tp, fp, tn, fn = cm['TP'], cm['FP'], cm['TN'], cm['FN']

        # Precision: TP / (TP + FP)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

        # Recall: TP / (TP + FN)
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        # F1: 2 * precision * recall / (precision + recall)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        prevention_rate = compute_prevention_rate(scores, labels, thresh)
        unnecessary_block_rate = compute_unnecessary_block_rate(scores, labels, thresh)

        results.append({
            'threshold': thresh,
            'auroc': compute_auroc(scores, labels),
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'prevention_rate': prevention_rate,
            'unnecessary_block_rate': unnecessary_block_rate
        })

    return pd.DataFrame(results)