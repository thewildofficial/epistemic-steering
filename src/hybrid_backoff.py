"""Hybrid backoff strategy for combining uncertainty signals.

Combines probe scores, MSP (maximum softmax probability), and entropy
into a single hybrid score with z-score normalization. Backs off from
probe to distributional signals when OOD alarm fires.

Functions:
- zscore_normalize: Z-score normalization with safe handling of edge cases
- compute_alpha: Compute probe weight (alpha) based on OOD alarm state
- hybrid_score: Combine probe, MSP, and entropy into a single score
"""

import numpy as np


def zscore_normalize(values: list[float]) -> list[float]:
    """Z-score normalize a list of values.

    When std is zero or length <= 1, returns all zeros.

    Args:
        values: List of float values to normalize.

    Returns:
        List of float z-scores (same length as input).
    """
    arr = np.array(values, dtype=float)
    n = len(arr)
    if n <= 1:
        return [0.0] * n
    std = float(np.std(arr))
    if std == 0.0:
        return [0.0] * n
    mean = float(np.mean(arr))
    return ((arr - mean) / std).tolist()


def compute_alpha(probe_score: float, ood_alarm: bool) -> float:
    """Compute alpha weight for the probe score in the hybrid mix.

    In-domain: alpha = 0.7 (trust the probe).
    When OOD alarm fires: alpha → 0.0 (back off to distributional signals).

    Args:
        probe_score: Probe confidence score (unused in alpha calc,
                     kept for API compatibility).
        ood_alarm: Whether the OOD detector alarm is active.

    Returns:
        Float alpha in [0.0, 0.7].
    """
    return 0.0 if ood_alarm else 0.7


def hybrid_score(
    probe_score: float,
    msp: float,
    entropy: float,
    alpha: float,
) -> float:
    """Compute hybrid score combining probe, MSP, and entropy.

    Formula:
        alpha * probe_norm
        + (1 - alpha) * (0.5 * msp_norm + 0.5 * entropy_norm)

    Where each signal is z-score normalized against itself.

    Args:
        probe_score: Raw probe score.
        msp: Maximum softmax probability.
        entropy: Entropy of the output distribution.
        alpha: Weight for the probe signal in [0, 1].

    Returns:
        Float hybrid score.
    """
    # Z-score normalize the three signals together
    scores = [probe_score, msp, entropy]
    probe_norm, msp_norm, entropy_norm = zscore_normalize(scores)

    return (
        alpha * probe_norm
        + (1.0 - alpha) * (0.5 * msp_norm + 0.5 * entropy_norm)
    )
