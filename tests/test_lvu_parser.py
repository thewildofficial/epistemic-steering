"""Tests for lvu_parser — hedge word detection and LVU scoring."""

import pytest
from src.lvu_parser import has_hedge_words, has_self_correction, lvu_score


class TestHasHedgeWords:
    def test_detects_maybe(self):
        assert has_hedge_words("Maybe we should try another approach")

    def test_detects_i_think(self):
        assert has_hedge_words("I think the answer could be 42")

    def test_detects_multiple_hedges(self):
        assert has_hedge_words("I'm not sure, but it might be possible")

    def test_no_hedge_on_confident_text(self):
        assert not has_hedge_words("The answer is 42.")

    def test_no_hedge_on_empty_text(self):
        assert not has_hedge_words("")

    def test_no_hedge_on_whitespace(self):
        assert not has_hedge_words("   ")


class TestHasSelfCorrection:
    def test_detects_wait(self):
        assert has_self_correction("Wait, that's not right")

    def test_detects_actually(self):
        assert has_self_correction("Actually, let me reconsider")

    def test_detects_hold_on(self):
        assert has_self_correction("Hold on, I need to check that")

    def test_no_correction_on_plain_text(self):
        assert not has_self_correction("The capital of France is Paris.")

    def test_no_correction_on_empty(self):
        assert not has_self_correction("")


class TestLvuScore:
    def test_hedge_words_contribute_half(self):
        score = lvu_score("I think this might be correct")
        assert score == pytest.approx(0.5)

    def test_self_correction_contributes_half(self):
        score = lvu_score("Wait, let me recalculate")
        assert score == pytest.approx(0.5)

    def test_both_contributions_sum_to_one(self):
        score = lvu_score("I think maybe the answer is... Wait, no.")
        assert score == pytest.approx(1.0)

    def test_confident_text_scores_zero(self):
        score = lvu_score("The answer is 42.")
        assert score == pytest.approx(0.0)

    def test_empty_text_scores_maximum(self):
        score = lvu_score("")
        assert score == pytest.approx(1.0)

    def test_whitespace_text_scores_maximum(self):
        score = lvu_score("   ")
        assert score == pytest.approx(1.0)
