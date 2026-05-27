"""OOD (Out-of-Distribution) detector using MSP gap.

Detects distributional shift by measuring the gap between the top-two
softmax probabilities. A large MSP gap indicates the model is confident
in one class (in-distribution). A small gap indicates uncertainty
in the prediction, potentially OOD.

Functions:
- detect_ood: Classify OOD alarm level from MSP gap
"""

ALARM_HIGH: str = "HIGH"
ALARM_MODERATE: str = "MODERATE"
ALARM_LOW: str = "LOW"


def detect_ood(msp_gap: float) -> str:
    """Detect OOD based on MSP gap.

    MSP gap = max(softmax) - second_max(softmax).

    Args:
        msp_gap: Gap between top and second MSP probabilities.
                 Must be in [0, 1].

    Returns:
        Alarm level: 'HIGH', 'MODERATE', or 'LOW'.
    """
    if msp_gap > 0.3:
        return ALARM_HIGH
    elif msp_gap > 0.1:
        return ALARM_MODERATE
    else:
        return ALARM_LOW
