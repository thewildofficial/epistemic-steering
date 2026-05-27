"""Tests for ood_detector — MSP-gap based OOD alarm levels."""

import pytest
from src.ood_detector import detect_ood, ALARM_HIGH, ALARM_MODERATE, ALARM_LOW


class TestDetectOod:
    def test_gap_above_0_3_triggers_high(self):
        assert detect_ood(0.5) == ALARM_HIGH
        assert detect_ood(0.31) == ALARM_HIGH
        assert detect_ood(1.0) == ALARM_HIGH

    def test_gap_0_0_triggers_low(self):
        assert detect_ood(0.0) == ALARM_LOW

    def test_small_gap_triggers_low(self):
        assert detect_ood(0.05) == ALARM_LOW
        assert detect_ood(0.1) == ALARM_LOW

    def test_moderate_gap_triggers_moderate(self):
        assert detect_ood(0.15) == ALARM_MODERATE
        assert detect_ood(0.2) == ALARM_MODERATE
        assert detect_ood(0.3) == ALARM_MODERATE

    def test_high_moderate_low_mutually_exclusive(self):
        results = {detect_ood(g) for g in [0.05, 0.2, 0.5]}
        assert results == {ALARM_LOW, ALARM_MODERATE, ALARM_HIGH}

    def test_edge_case_0_3_is_moderate_not_high(self):
        assert detect_ood(0.3) == ALARM_MODERATE
