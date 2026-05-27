"""LVU (Linguistic Verbal Uncertainty) parser.

Analyzes generated text for linguistic uncertainty cues across four categories:
hedging, self-correction, confidence gaps, and overconfidence markers.

Classes:
- LVUParser: Full parser with 4-category weighted scoring, returns dict.

Functions (backward compatible with T1 tests):
- has_hedge_words: Check presence of hedge words in text
- has_self_correction: Check presence of self-correction patterns
- lvu_score: Simple binary score (hedge + self_correction only)
"""

import re

# === FROZEN LEXICON — NEVER MODIFIED AFTER CREATION ===
# Per the experiment plan: these constants are pre-registered.
# No additions, no removals, no runtime modifications.

HEDGING: list[str] = [
    "maybe", "perhaps", "possibly", "could", "might",
    "i think", "i believe", "in my opinion", "presumably",
    "it seems", "approximately", "roughly", "about",
]

SELF_CORRECTION: list[str] = [
    "wait,", "actually,", "correction:",
    "let me reconsider", "on second thought",
    "i made a mistake", "i was wrong",
]

CONFIDENCE_GAP: list[str] = [
    "i'm not sure", "i don't know", " uncertain", "not certain",
]

OVERCONFIDENCE_INVERSE: list[str] = [
    "definitely correct", "this will work",
    "absolutely certain", "without a doubt",
]

# Pre-registered weights for overall score
WEIGHTS = {
    "hedging": 0.4,
    "self_correction": 0.3,
    "confidence_gap": 0.2,
    "overconfidence": 0.1,
}


def _count_lexicon_hits(text: str, lexicon: list[str]) -> int:
    """Count how many lexicon entries appear in the text."""
    text_lower = text.lower().strip()
    if not text_lower:
        return 0
    text_clean = re.sub(r"[^\w\s]", " ", text_lower)
    # Strip punctuation from lexicon terms too, since text is stripped
    cleaned_lexicon = [re.sub(r"[^\w\s]", " ", t).strip() for t in lexicon]
    return sum(1 for term in cleaned_lexicon if term in text_clean)


def _score_category(text: str, lexicon: list[str]) -> float:
    """Score a single category: count(lexicon hits) / max(1, token_count)."""
    token_count = max(1, len(text.strip().split()))
    hits = _count_lexicon_hits(text, lexicon)
    return min(hits / token_count, 1.0)


class LVUParser:
    """Linguistic Verbal Uncertainty parser with 4-category weighted scoring.

    Usage:
        parser = LVUParser()
        result = parser.score("I think the answer is 42. Wait, actually...")
        print(result["overall_score"])  # weighted combination
        print(result["hedging_score"])  # hedging category only
    """

    def score(self, text: str) -> dict:
        """Compute LVU scores across all four categories.

        Args:
            text: Generated text to analyze.

        Returns:
            dict with keys:
                hedging_score, self_correction_score,
                confidence_gap_score, overconfidence_score, overall_score
            All values in [0, 1]. Empty text returns all zeros.
        """
        if not text or not text.strip():
            return {
                "hedging_score": 0.0,
                "self_correction_score": 0.0,
                "confidence_gap_score": 0.0,
                "overconfidence_score": 0.0,
                "overall_score": 0.0,
            }

        hedging = _score_category(text, HEDGING)
        self_correction = _score_category(text, SELF_CORRECTION)
        confidence_gap = _score_category(text, CONFIDENCE_GAP)
        overconfidence = _score_category(text, OVERCONFIDENCE_INVERSE)

        overall = (
            hedging * WEIGHTS["hedging"]
            + self_correction * WEIGHTS["self_correction"]
            + confidence_gap * WEIGHTS["confidence_gap"]
            + overconfidence * WEIGHTS["overconfidence"]
        )

        return {
            "hedging_score": hedging,
            "self_correction_score": self_correction,
            "confidence_gap_score": confidence_gap,
            "overconfidence_score": overconfidence,
            "overall_score": round(overall, 4),
        }


# === BACKWARD-COMPATIBLE FUNCTIONS (T1 test interface) ===

# Extended hedge list for backward compat includes confidence gap terms
# that the T1 tests expect but the plan does NOT include in the frozen HEDGING set.
_COMPAT_HEDGE: list[str] = HEDGING + [
    "probably", "i'm not sure", "i'm uncertain", "i don't know",
    "could be", "might be", "not sure", "not certain",
    "sort of", "kind of", "i guess", "i suppose",
    "it appears", "it might",
]


def has_hedge_words(text: str) -> bool:
    """Check if text contains hedge words (backward-compatible, broader lexicon)."""
    if not text or not text.strip():
        return False
    text_lower = text.lower()
    return any(hedge in text_lower for hedge in _COMPAT_HEDGE)


def has_self_correction(text: str) -> bool:
    """Check if text contains self-correction patterns (backward-compatible)."""
    if not text or not text.strip():
        return False
    text_lower = text.lower()
    patterns = [
        r"wait", r"actually,?", r"on second thought", r"let me reconsider",
        r"let me think", r"hold on", r"i mean", r"correction:",
        r"no,", r"that's not right", r"that doesn't sound right",
        r"let me check", r"let me verify", r"i take that back",
        r"hmm", r"hmm,", r"well,",
    ]
    return any(re.search(pat, text_lower) for pat in patterns)


def lvu_score(text: str) -> float:
    """Compute simple LVU score [0, 1] (backward-compatible with T1 tests).

    Empty/whitespace text → 1.0 (maximum uncertainty).
    Hedge words → +0.5, self-correction → +0.5.
    """
    if not text or not text.strip():
        return 1.0
    score = 0.0
    if has_hedge_words(text):
        score += 0.5
    if has_self_correction(text):
        score += 0.5
    return min(score, 1.0)
