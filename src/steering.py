"""Epistemic steering system for LLM hallucination prevention.

Routes questions to direct answer, CoT with monitoring, or abstention
based on epistemic confidence from probe scores.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
import numpy as np

from probe import compute_confidence


@dataclass
class SteeringResult:
    """Result of a steering decision."""
    question_id: str
    dataset: str
    route: str
    prefill_confidence: float
    final_answer: Optional[str] = None
    abstained: bool = False
    tokens_used: int = 0
    generation_trace: List[Dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dict for JSON serialization."""
        return {
            "question_id": self.question_id,
            "dataset": self.dataset,
            "route": self.route,
            "prefill_confidence": self.prefill_confidence,
            "final_answer": self.final_answer,
            "abstained": self.abstained,
            "tokens_used": self.tokens_used,
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        import json
        return json.dumps(self.to_dict())


class PrefillProbeRouter:
    """Routes based on prefill probe confidence score.

    Args:
        probe_weights: dict with 'coef' (np.ndarray) and 'intercept' (float)
        threshold_high: float, score >= this -> direct answer
        threshold_low: float, score <= this -> abstain
                       score between -> CoT with monitoring
    """

    def __init__(self, probe_weights: dict, threshold_high: float = 0.7, threshold_low: float = 0.3):
        self.weights = probe_weights
        self.threshold_high = threshold_high
        self.threshold_low = threshold_low

    def compute_confidence(self, activation: np.ndarray) -> float:
        """Compute prefill probe confidence: sigma(w^T h + b)"""
        return compute_confidence(activation, self.weights)

    def route(self, question_id: str, activation: np.ndarray) -> str:
        """Returns 'direct', 'cot', or 'abstain'."""
        score = self.compute_confidence(activation)
        if score >= self.threshold_high:
            return "direct"
        elif score <= self.threshold_low:
            return "abstain"
        else:
            return "cot"


class GenerationTimeMonitor:
    """Monitors confidence during CoT generation.

    Checks probe confidence at every 5th token during generation.
    If confidence drops below threshold, triggers abstention.

    Args:
        gen_time_probe_weights: dict with 'coef' and 'intercept'
        abstain_threshold: float, score < this -> trigger abstention
        check_every_n_tokens: int, check every N tokens (default 5)
    """

    def __init__(self, gen_time_probe_weights: dict, abstain_threshold: float = 0.3, check_every_n_tokens: int = 5):
        self.weights = gen_time_probe_weights
        self.abstain_threshold = abstain_threshold
        self.check_every_n = check_every_n_tokens

    def check(self, token_position: int, activation: np.ndarray) -> str:
        """Returns 'continue' or 'abstain'."""
        if token_position % self.check_every_n != 0:
            return "continue"

        score = compute_confidence(activation, self.weights)
        if score < self.abstain_threshold:
            return "abstain"
        return "continue"


class EpistemicSteeringSystem:
    """Main steering system combining prefill routing + generation-time monitoring.

    Pipeline:
    1. Prefill probe -> route (direct / CoT / abstain)
    2. If direct: generate answer immediately
    3. If CoT: generate with generation-time monitoring
    4. If abstain: return "I don't know"

    Args:
        prefill_router: PrefillProbeRouter instance
        gen_time_monitor: GenerationTimeMonitor instance (optional)
    """

    ABSTAIN_ANSWER = "I don't know"

    def __init__(self, prefill_router: PrefillProbeRouter, gen_time_monitor: Optional[GenerationTimeMonitor] = None):
        self.prefill_router = prefill_router
        self.gen_time_monitor = gen_time_monitor

    @staticmethod
    def extract_answer(cot_output: str) -> str:
        """Extract final answer from CoT output.

        Handles edge cases:
        - "?" answer -> abstention
        - unparseable output -> abstention
        - empty output -> abstention
        """
        if not cot_output or not cot_output.strip():
            return EpistemicSteeringSystem.ABSTAIN_ANSWER

        answer = cot_output.strip()

        if answer == "?":
            return EpistemicSteeringSystem.ABSTAIN_ANSWER

        return answer

    @staticmethod
    def finalize_answer(result: SteeringResult, generated_text: Optional[str]) -> SteeringResult:
        """Finalize answer for a steering result.

        Handles:
        - "?" answers -> abstention
        - unparseable CoT -> "I don't know"
        - empty generation -> "I don't know"
        """
        if result.route == "abstain":
            result.final_answer = EpistemicSteeringSystem.ABSTAIN_ANSWER
            result.abstained = True
            return result

        if not generated_text or not generated_text.strip():
            result.final_answer = EpistemicSteeringSystem.ABSTAIN_ANSWER
            result.abstained = True
            return result

        extracted = EpistemicSteeringSystem.extract_answer(generated_text)
        if extracted == EpistemicSteeringSystem.ABSTAIN_ANSWER:
            result.final_answer = EpistemicSteeringSystem.ABSTAIN_ANSWER
            result.abstained = True
        else:
            result.final_answer = extracted
            result.abstained = False

        return result

    def route_question(self, question_id: str, dataset: str, prefill_activation: np.ndarray) -> SteeringResult:
        """Route a single question through the steering pipeline.

        Returns SteeringResult with route decision.
        Does NOT actually generate text -- just makes the routing decision.
        """
        confidence = self.prefill_router.compute_confidence(prefill_activation)
        route = self.prefill_router.route(question_id, prefill_activation)

        result = SteeringResult(
            question_id=question_id,
            dataset=dataset,
            route=route,
            prefill_confidence=confidence,
        )

        if route == "abstain":
            result.final_answer = self.ABSTAIN_ANSWER
            result.abstained = True
        elif route == "direct":
            result.final_answer = None
            result.abstained = False
        elif route == "cot":
            result.abstained = False

        return result

    def batch_route(self, questions: List[tuple]) -> List[SteeringResult]:
        """Route multiple questions, returns list of SteeringResults.

        Args:
            questions: List of tuples (question_id, dataset, prefill_activation)

        Returns:
            List of SteeringResult objects.
        """
        results = []
        for qid, dataset, activation in questions:
            result = self.route_question(qid, dataset, activation)
            results.append(result)
        return results

    def get_statistics(self, results: List[SteeringResult]) -> dict:
        """Compute routing statistics: % direct, % cot, % abstain, etc."""
        if not results:
            return {
                "total": 0,
                "direct_count": 0,
                "cot_count": 0,
                "abstain_count": 0,
                "direct_pct": 0.0,
                "cot_pct": 0.0,
                "abstain_pct": 0.0,
                "mean_prefill_confidence": 0.0,
            }

        total = len(results)
        direct_count = sum(1 for r in results if r.route == "direct")
        cot_count = sum(1 for r in results if r.route == "cot")
        abstain_count = sum(1 for r in results if r.route == "abstain")

        confidences = [r.prefill_confidence for r in results]
        mean_conf = float(np.mean(confidences)) if confidences else 0.0

        return {
            "total": total,
            "direct_count": direct_count,
            "cot_count": cot_count,
            "abstain_count": abstain_count,
            "direct_pct": direct_count / total,
            "cot_pct": cot_count / total,
            "abstain_pct": abstain_count / total,
            "mean_prefill_confidence": mean_conf,
        }


@dataclass
class BatchSteeringResult:
    """Container for multiple steering results with aggregated statistics."""
    results: List[SteeringResult]

    def summary(self) -> dict:
        """Compute summary statistics across all results."""
        if not self.results:
            return {
                "total": 0,
                "direct_count": 0,
                "cot_count": 0,
                "abstain_count": 0,
                "direct_pct": 0.0,
                "cot_pct": 0.0,
                "abstain_pct": 0.0,
                "avg_confidence": 0.0,
                "avg_tokens": 0.0,
            }

        total = len(self.results)
        direct_count = sum(1 for r in self.results if r.route == "direct")
        cot_count = sum(1 for r in self.results if r.route == "cot")
        abstain_count = sum(1 for r in self.results if r.route == "abstain")

        confidences = [r.prefill_confidence for r in self.results]
        avg_confidence = float(np.mean(confidences)) if confidences else 0.0

        tokens = [r.tokens_used for r in self.results]
        avg_tokens = float(np.mean(tokens)) if tokens else 0.0

        return {
            "total": total,
            "direct_count": direct_count,
            "cot_count": cot_count,
            "abstain_count": abstain_count,
            "direct_pct": direct_count / total,
            "cot_pct": cot_count / total,
            "abstain_pct": abstain_count / total,
            "avg_confidence": avg_confidence,
            "avg_tokens": avg_tokens,
        }
