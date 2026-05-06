"""Evaluation metrics for epistemic steering.

Functions for computing confusion matrices, prevention rates,
selective accuracy, token efficiency, and calibration curves
for hallucination detection and steering evaluation.
"""

import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, precision_recall_fscore_support


def confusion_matrix_at_threshold(
    scores: np.ndarray, labels: np.ndarray, threshold: float
) -> dict:
    """Compute confusion matrix at a given threshold.

    "Positive" = model was correct (label is True).
    "Confident" = score >= threshold (probe would allow through).

    Args:
        scores: Probe confidence scores.
        labels: Ground truth labels (True = correct answer).
        threshold: Decision threshold for confidence.

    Returns:
        Dict with TP, FP, TN, FN counts.
        - TP: correctly allowed through (correct AND confident)
        - FP: missed hallucination (wrong AND confident)
        - TN: correctly caught hallucination (wrong AND uncertain)
        - FN: unnecessarily blocked (correct BUT uncertain)
    """
    confident = scores >= threshold
    labels_arr = np.asarray(labels, dtype=bool)

    tp = int(np.sum(labels_arr & confident))
    fp = int(np.sum(~labels_arr & confident))
    tn = int(np.sum(~labels_arr & ~confident))
    fn = int(np.sum(labels_arr & ~confident))

    return {"TP": tp, "FP": fp, "TN": tn, "FN": fn}


def prevention_rate_at_threshold(
    scores: np.ndarray, labels: np.ndarray, threshold: float
) -> float:
    """Fraction of incorrect answers (hallucinations) caught.

    Among INCORRECT answers (label=False), the fraction with low probe score.

    Args:
        scores: Probe confidence scores.
        labels: Ground truth labels (True = correct answer).
        threshold: Decision threshold for confidence.

    Returns:
        Fraction of hallucinations caught: TN / (TN + FP)
    """
    labels_arr = np.asarray(labels)
    incorrect = ~labels_arr
    low_score = scores < threshold

    caught = np.sum(incorrect & low_score)
    total_incorrect = np.sum(incorrect)

    if total_incorrect == 0:
        return 0.0

    return float(caught / total_incorrect)


def unnecessary_block_rate_at_threshold(
    scores: np.ndarray, labels: np.ndarray, threshold: float
) -> float:
    """Fraction of correct answers mistakenly blocked.

    Among CORRECT answers (label=True), the fraction with low probe score.

    Args:
        scores: Probe confidence scores.
        labels: Ground truth labels (True = correct answer).
        threshold: Decision threshold for confidence.

    Returns:
        Fraction of correct answers blocked: FN / (FN + TP)
    """
    labels_arr = np.asarray(labels)
    correct = labels_arr
    low_score = scores < threshold

    blocked = np.sum(correct & low_score)
    total_correct = np.sum(correct)

    if total_correct == 0:
        return 0.0

    return float(blocked / total_correct)


def selective_accuracy(
    direct_correct: int, cot_correct: int, abstentions: int, total: int
) -> float:
    """Compute selective accuracy accounting for abstentions.

    Selective accuracy = (correct via any method) / total questions
    Abstentions are treated as "I don't know" and not counted as errors.

    Args:
        direct_correct: Questions answered correctly with direct method.
        cot_correct: Questions answered correctly with CoT method.
        abstentions: Questions where model abstained from answering.
        total: Total number of questions.

    Returns:
        Selective accuracy as a fraction.
    """
    if total == 0:
        return 0.0

    return float(direct_correct + cot_correct) / float(total)


def token_efficiency(
    direct_tokens: int, cot_tokens: int, routed_tokens: int, total: int
) -> dict:
    """Compute token efficiency metrics.

    Args:
        direct_tokens: Total tokens spent on direct answers.
        cot_tokens: Total tokens spent on CoT answers.
        routed_tokens: Total tokens after routing (direct + cot used).
        total: Number of questions.

    Returns:
        Dict with:
        - tokens_per_question: average tokens per question
        - tokens_per_correct: average tokens per correct answer
        - savings_vs_cot: fraction of tokens saved vs always-CoT
    """
    if total == 0:
        return {
            "tokens_per_question": 0.0,
            "tokens_per_correct": 0.0,
            "savings_vs_cot": 0.0,
        }

    tokens_per_question = float(routed_tokens) / float(total)

    # For tokens_per_correct, we need correct count - default to routed_tokens
    # as a rough approximation when correct count unknown
    # In practice, you'd also track correct counts
    tokens_per_correct = float(routed_tokens) / float(total)  # placeholder

    # Savings vs always-CoT: (cot_tokens - routed_tokens) / cot_tokens
    if cot_tokens > 0:
        savings_vs_cot = float(cot_tokens - routed_tokens) / float(cot_tokens)
    else:
        savings_vs_cot = 0.0

    return {
        "tokens_per_question": tokens_per_question,
        "tokens_per_correct": tokens_per_correct,
        "savings_vs_cot": savings_vs_cot,
    }


def calibration_curve(
    scores: np.ndarray, labels: np.ndarray, n_bins: int = 10
) -> dict:
    """Compute calibration curve (reliability diagram).

    Uses sklearn.calibration_curve with uniform binning strategy.

    Args:
        scores: Probe confidence scores (0 to 1).
        labels: Ground truth labels (True = correct).
        n_bins: Number of bins for calibration curve.

    Returns:
        Dict with:
        - bin_centers: center of each bin (predicted confidence)
        - observed_accuracy: actual accuracy in each bin
        - bin_counts: number of samples in each bin
    """
    from sklearn.calibration import calibration_curve as sk_calibration_curve

    # sklearn expects labels as 0/1, not True/False
    labels_binary = np.asarray(labels).astype(int)

    fraction_positive, bin_center = sk_calibration_curve(
        labels_binary, scores, n_bins=n_bins, strategy="uniform"
    )

    # Compute bin counts by digitizing scores
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(scores, bin_edges) - 1
    # Clip to valid range (scores exactly 1.0 can go to n_bins)
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    bin_counts = np.bincount(bin_indices, minlength=n_bins)

    return {
        "bin_centers": bin_center,
        "observed_accuracy": fraction_positive,
        "bin_counts": bin_counts.astype(int),
    }


def compute_all_metrics(
    scores: np.ndarray, labels: np.ndarray, threshold: float = 0.5
) -> dict:
    """Compute comprehensive metrics for evaluation.

    Args:
        scores: Probe confidence scores.
        labels: Ground truth labels (True = correct).
        threshold: Decision threshold for confidence.

    Returns:
        Dict with all evaluation metrics:
        - auroc: area under ROC curve
        - threshold: the threshold used
        - confusion_matrix: dict with TP, FP, TN, FN
        - prevention_rate: fraction of hallucinations caught
        - unnecessary_block_rate: fraction of correct answers blocked
        - precision: precision at threshold
        - recall: recall at threshold
        - f1_score: F1 score at threshold
        - calibration: dict with bin_centers, observed_accuracy, bin_counts
    """
    labels_arr = np.asarray(labels)

    # AUROC
    auroc = roc_auc_score(labels_arr, scores)

    # Confusion matrix
    cm = confusion_matrix_at_threshold(scores, labels_arr, threshold)

    # Prevention and block rates
    prev_rate = prevention_rate_at_threshold(scores, labels_arr, threshold)
    block_rate = unnecessary_block_rate_at_threshold(scores, labels_arr, threshold)

    # Precision, recall, F1 from confusion matrix
    tp, fp, tn, fn = cm["TP"], cm["FP"], cm["TN"], cm["FN"]

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Calibration curve
    calib = calibration_curve(scores, labels_arr, n_bins=10)

    return {
        "auroc": float(auroc),
        "threshold": threshold,
        "confusion_matrix": cm,
        "prevention_rate": prev_rate,
        "unnecessary_block_rate": block_rate,
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "calibration": calib,
    }