import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from steering import (
    EpistemicSteeringSystem,
    GenerationTimeMonitor,
    PrefillProbeRouter,
    SteeringResult,
    BatchSteeringResult,
)


class TestSteeringResult:
    def test_default_fields(self):
        result = SteeringResult(
            question_id="q1",
            dataset="mmlu",
            route="direct",
            prefill_confidence=0.8,
        )
        assert result.question_id == "q1"
        assert result.dataset == "mmlu"
        assert result.route == "direct"
        assert result.prefill_confidence == pytest.approx(0.8)
        assert result.final_answer is None
        assert result.abstained is False
        assert result.tokens_used == 0
        assert result.generation_trace == []


class TestSteeringResultSerialization:
    def test_to_dict(self):
        result = SteeringResult(
            question_id="q1",
            dataset="mmlu",
            route="direct",
            prefill_confidence=0.8,
            final_answer="42",
            abstained=False,
            tokens_used=100,
        )
        d = result.to_dict()
        assert d["question_id"] == "q1"
        assert d["dataset"] == "mmlu"
        assert d["route"] == "direct"
        assert d["prefill_confidence"] == 0.8
        assert d["final_answer"] == "42"
        assert d["abstained"] is False
        assert d["tokens_used"] == 100

    def test_to_json(self):
        result = SteeringResult(
            question_id="q1",
            dataset="mmlu",
            route="cot",
            prefill_confidence=0.5,
            final_answer="42",
            abstained=False,
            tokens_used=50,
        )
        import json
        j = result.to_json()
        parsed = json.loads(j)
        assert parsed["question_id"] == "q1"
        assert parsed["route"] == "cot"
        assert parsed["final_answer"] == "42"


class TestBatchSteeringResult:
    def make_weights(self, intercept: float = 0.0) -> dict:
        return {'coef': np.zeros(2560), 'intercept': intercept}

    def test_summary_empty(self):
        batch = BatchSteeringResult(results=[])
        s = batch.summary()
        assert s["total"] == 0
        assert s["direct_pct"] == 0.0
        assert s["avg_confidence"] == 0.0

    def test_summary_all_direct(self):
        results = [
            SteeringResult("q1", "mmlu", "direct", 0.8, "a", False, 10),
            SteeringResult("q2", "mmlu", "direct", 0.9, "b", False, 20),
        ]
        batch = BatchSteeringResult(results=results)
        s = batch.summary()
        assert s["total"] == 2
        assert s["direct_count"] == 2
        assert s["direct_pct"] == 1.0
        assert s["cot_count"] == 0
        assert s["abstain_count"] == 0
        assert s["avg_confidence"] == pytest.approx(0.85)
        assert s["avg_tokens"] == pytest.approx(15.0)

    def test_summary_mixed_routes(self):
        results = [
            SteeringResult("q1", "mmlu", "direct", 0.8, "a", False, 10),
            SteeringResult("q2", "mmlu", "cot", 0.5, "b", False, 50),
            SteeringResult("q3", "mmlu", "abstain", 0.2, "I don't know", True, 5),
        ]
        batch = BatchSteeringResult(results=results)
        s = batch.summary()
        assert s["total"] == 3
        assert s["direct_count"] == 1
        assert s["cot_count"] == 1
        assert s["abstain_count"] == 1
        assert s["direct_pct"] == pytest.approx(1/3)
        assert s["cot_pct"] == pytest.approx(1/3)
        assert s["abstain_pct"] == pytest.approx(1/3)

    def test_custom_fields(self):
        trace = [{"token": 5, "action": "continue"}]
        result = SteeringResult(
            question_id="q2",
            dataset="gsm8k",
            route="cot",
            prefill_confidence=0.5,
            final_answer="42",
            abstained=False,
            tokens_used=100,
            generation_trace=trace,
        )
        assert result.final_answer == "42"
        assert result.tokens_used == 100
        assert result.generation_trace == trace


class TestPrefillProbeRouter:
    def make_weights(self, intercept: float = 0.0) -> dict:
        return {'coef': np.zeros(2560), 'intercept': intercept}

    def test_route_direct(self):
        weights = self.make_weights(intercept=10.0)
        router = PrefillProbeRouter(weights, threshold_high=0.7, threshold_low=0.3)
        activation = np.zeros(2560)
        assert router.route("q1", activation) == "direct"

    def test_route_abstain(self):
        weights = self.make_weights(intercept=-10.0)
        router = PrefillProbeRouter(weights, threshold_high=0.7, threshold_low=0.3)
        activation = np.zeros(2560)
        assert router.route("q1", activation) == "abstain"

    def test_route_cot(self):
        weights = self.make_weights(intercept=0.0)
        router = PrefillProbeRouter(weights, threshold_high=0.7, threshold_low=0.3)
        activation = np.zeros(2560)
        assert router.route("q1", activation) == "cot"

    def test_route_at_high_boundary(self):
        weights = self.make_weights(intercept=0.0)
        router = PrefillProbeRouter(weights, threshold_high=0.5, threshold_low=0.2)
        activation = np.zeros(2560)
        assert router.compute_confidence(activation) == pytest.approx(0.5)
        assert router.route("q1", activation) == "direct"

    def test_route_at_low_boundary(self):
        weights = self.make_weights(intercept=0.0)
        router = PrefillProbeRouter(weights, threshold_high=0.8, threshold_low=0.5)
        activation = np.zeros(2560)
        assert router.compute_confidence(activation) == pytest.approx(0.5)
        assert router.route("q1", activation) == "abstain"

    def test_compute_confidence_range(self):
        weights = self.make_weights()
        router = PrefillProbeRouter(weights)
        activation = np.random.randn(2560)
        conf = router.compute_confidence(activation)
        assert 0.0 <= conf <= 1.0

    def test_threshold_zero(self):
        weights = self.make_weights(intercept=10.0)
        router = PrefillProbeRouter(weights, threshold_high=0.0, threshold_low=0.0)
        activation = np.zeros(2560)
        assert router.route("q1", activation) == "direct"

    def test_threshold_one(self):
        weights = self.make_weights(intercept=-10.0)
        router = PrefillProbeRouter(weights, threshold_high=1.0, threshold_low=1.0)
        activation = np.zeros(2560)
        assert router.route("q1", activation) == "abstain"

    def test_empty_activation_raises(self):
        weights = self.make_weights()
        router = PrefillProbeRouter(weights)
        with pytest.raises(Exception):
            router.route("q1", np.array([]))


class TestGenerationTimeMonitor:
    def make_weights(self, intercept: float = 0.0) -> dict:
        return {'coef': np.zeros(2560), 'intercept': intercept}

    def test_check_every_n_skips(self):
        weights = self.make_weights(intercept=-10.0)
        monitor = GenerationTimeMonitor(weights, abstain_threshold=0.3, check_every_n_tokens=5)
        activation = np.zeros(2560)
        assert monitor.check(1, activation) == "continue"
        assert monitor.check(2, activation) == "continue"
        assert monitor.check(3, activation) == "continue"
        assert monitor.check(4, activation) == "continue"

    def test_check_abstain_on_low_confidence(self):
        weights = self.make_weights(intercept=-10.0)
        monitor = GenerationTimeMonitor(weights, abstain_threshold=0.3, check_every_n_tokens=5)
        activation = np.zeros(2560)
        assert monitor.check(5, activation) == "abstain"
        assert monitor.check(10, activation) == "abstain"

    def test_check_continue_on_high_confidence(self):
        weights = self.make_weights(intercept=10.0)
        monitor = GenerationTimeMonitor(weights, abstain_threshold=0.3, check_every_n_tokens=5)
        activation = np.zeros(2560)
        assert monitor.check(5, activation) == "continue"
        assert monitor.check(10, activation) == "continue"

    def test_check_position_zero(self):
        weights = self.make_weights(intercept=-10.0)
        monitor = GenerationTimeMonitor(weights, abstain_threshold=0.3, check_every_n_tokens=5)
        activation = np.zeros(2560)
        assert monitor.check(0, activation) == "abstain"

    def test_custom_check_every_n(self):
        weights = self.make_weights(intercept=-10.0)
        monitor = GenerationTimeMonitor(weights, abstain_threshold=0.3, check_every_n_tokens=3)
        activation = np.zeros(2560)
        assert monitor.check(3, activation) == "abstain"
        assert monitor.check(6, activation) == "abstain"
        assert monitor.check(1, activation) == "continue"


class TestEpistemicSteeringSystem:
    def make_weights(self, intercept: float = 0.0) -> dict:
        return {'coef': np.zeros(2560), 'intercept': intercept}

    def test_route_question_direct(self):
        weights = self.make_weights(intercept=10.0)
        router = PrefillProbeRouter(weights, threshold_high=0.7, threshold_low=0.3)
        system = EpistemicSteeringSystem(router)
        activation = np.zeros(2560)
        result = system.route_question("q1", "mmlu", activation)
        assert result.route == "direct"
        assert result.abstained is False
        assert result.final_answer is None

    def test_route_question_abstain(self):
        weights = self.make_weights(intercept=-10.0)
        router = PrefillProbeRouter(weights, threshold_high=0.7, threshold_low=0.3)
        system = EpistemicSteeringSystem(router)
        activation = np.zeros(2560)
        result = system.route_question("q1", "mmlu", activation)
        assert result.route == "abstain"
        assert result.abstained is True
        assert result.final_answer == "I don't know"

    def test_route_question_cot(self):
        weights = self.make_weights(intercept=0.0)
        router = PrefillProbeRouter(weights, threshold_high=0.7, threshold_low=0.3)
        system = EpistemicSteeringSystem(router)
        activation = np.zeros(2560)
        result = system.route_question("q1", "gsm8k", activation)
        assert result.route == "cot"
        assert result.abstained is False

    def test_batch_route(self):
        weights = self.make_weights(intercept=0.0)
        router = PrefillProbeRouter(weights, threshold_high=0.7, threshold_low=0.3)
        system = EpistemicSteeringSystem(router)
        questions = [
            ("q1", "mmlu", np.zeros(2560)),
            ("q2", "gsm8k", np.zeros(2560)),
        ]
        results = system.batch_route(questions)
        assert len(results) == 2
        assert all(isinstance(r, SteeringResult) for r in results)
        assert results[0].question_id == "q1"
        assert results[1].question_id == "q2"

    def test_get_statistics(self):
        weights = self.make_weights(intercept=0.0)
        router = PrefillProbeRouter(weights, threshold_high=0.7, threshold_low=0.3)
        system = EpistemicSteeringSystem(router)
        results = [
            system.route_question("q1", "mmlu", np.zeros(2560)),
            system.route_question("q2", "mmlu", np.zeros(2560)),
            system.route_question("q3", "gsm8k", np.zeros(2560)),
        ]
        stats = system.get_statistics(results)
        assert stats['total'] == 3
        assert stats['cot_count'] == 3
        assert stats['cot_pct'] == pytest.approx(1.0)
        assert stats['direct_count'] == 0
        assert stats['abstain_count'] == 0
        assert 'mean_prefill_confidence' in stats

    def test_get_statistics_empty(self):
        weights = self.make_weights()
        router = PrefillProbeRouter(weights)
        system = EpistemicSteeringSystem(router)
        stats = system.get_statistics([])
        assert stats['total'] == 0
        assert stats['direct_pct'] == 0.0
        assert stats['cot_pct'] == 0.0
        assert stats['abstain_pct'] == 0.0

    def test_get_statistics_mixed_routes(self):
        high_weights = self.make_weights(intercept=10.0)
        low_weights = self.make_weights(intercept=-10.0)
        mid_weights = self.make_weights(intercept=0.0)

        router_high = PrefillProbeRouter(high_weights, threshold_high=0.7, threshold_low=0.3)
        router_low = PrefillProbeRouter(low_weights, threshold_high=0.7, threshold_low=0.3)
        router_mid = PrefillProbeRouter(mid_weights, threshold_high=0.7, threshold_low=0.3)

        system_high = EpistemicSteeringSystem(router_high)
        system_low = EpistemicSteeringSystem(router_low)
        system_mid = EpistemicSteeringSystem(router_mid)

        results = [
            system_high.route_question("q1", "mmlu", np.zeros(2560)),
            system_mid.route_question("q2", "mmlu", np.zeros(2560)),
            system_low.route_question("q3", "gsm8k", np.zeros(2560)),
        ]

        stats = EpistemicSteeringSystem(router_mid).get_statistics(results)
        assert stats['direct_count'] == 1
        assert stats['cot_count'] == 1
        assert stats['abstain_count'] == 1
        assert stats['direct_pct'] == pytest.approx(1 / 3)
        assert stats['cot_pct'] == pytest.approx(1 / 3)
        assert stats['abstain_pct'] == pytest.approx(1 / 3)

    def test_with_gen_time_monitor(self):
        prefill_weights = self.make_weights(intercept=0.0)
        gen_weights = self.make_weights(intercept=10.0)
        router = PrefillProbeRouter(prefill_weights, threshold_high=0.7, threshold_low=0.3)
        monitor = GenerationTimeMonitor(gen_weights, abstain_threshold=0.3, check_every_n_tokens=5)
        system = EpistemicSteeringSystem(router, monitor)
        assert system.gen_time_monitor is not None

    def test_prefill_confidence_in_result(self):
        weights = self.make_weights(intercept=0.0)
        router = PrefillProbeRouter(weights)
        system = EpistemicSteeringSystem(router)
        activation = np.zeros(2560)
        result = system.route_question("q1", "mmlu", activation)
        assert result.prefill_confidence == pytest.approx(0.5)


class TestEdgeCases:
    def test_nonstandard_activation_shape(self):
        weights = {'coef': np.zeros(128), 'intercept': 0.0}
        router = PrefillProbeRouter(weights)
        activation = np.zeros(128)
        assert router.compute_confidence(activation) == pytest.approx(0.5)

    def test_very_high_threshold(self):
        weights = {'coef': np.zeros(2560), 'intercept': 0.0}
        router = PrefillProbeRouter(weights, threshold_high=0.99, threshold_low=0.98)
        activation = np.zeros(2560)
        assert router.route("q1", activation) == "abstain"

    def test_very_low_threshold(self):
        weights = {'coef': np.zeros(2560), 'intercept': 0.0}
        router = PrefillProbeRouter(weights, threshold_high=0.02, threshold_low=0.01)
        activation = np.zeros(2560)
        assert router.route("q1", activation) == "direct"

    def test_equal_thresholds(self):
        weights = {'coef': np.zeros(2560), 'intercept': 0.0}
        router = PrefillProbeRouter(weights, threshold_high=0.5, threshold_low=0.5)
        activation = np.zeros(2560)
        assert router.route("q1", activation) == "direct"

    def test_gen_monitor_threshold_zero(self):
        weights = {'coef': np.zeros(2560), 'intercept': 0.0}
        monitor = GenerationTimeMonitor(weights, abstain_threshold=0.0, check_every_n_tokens=1)
        activation = np.zeros(2560)
        assert monitor.check(1, activation) == "continue"

    def test_gen_monitor_threshold_one(self):
        weights = {'coef': np.zeros(2560), 'intercept': 0.0}
        monitor = GenerationTimeMonitor(weights, abstain_threshold=1.0, check_every_n_tokens=1)
        activation = np.zeros(2560)
        assert monitor.check(1, activation) == "abstain"


class TestEdgeCaseHandling:
    def make_weights(self, intercept: float = 0.0) -> dict:
        return {'coef': np.zeros(2560), 'intercept': intercept}

    def test_question_mark_answer_abstains(self):
        weights = self.make_weights(intercept=0.0)
        router = PrefillProbeRouter(weights, threshold_high=0.7, threshold_low=0.3)
        system = EpistemicSteeringSystem(router)
        result = SteeringResult("q1", "mmlu", "cot", 0.5)
        finalized = system.finalize_answer(result, "?")
        assert finalized.final_answer == "I don't know"
        assert finalized.abstained is True

    def test_unparseable_cot_abstains(self):
        weights = self.make_weights(intercept=0.0)
        router = PrefillProbeRouter(weights, threshold_high=0.7, threshold_low=0.3)
        system = EpistemicSteeringSystem(router)
        result = SteeringResult("q1", "mmlu", "cot", 0.5)
        finalized = system.finalize_answer(result, "")
        assert finalized.final_answer == "I don't know"
        assert finalized.abstained is True

    def test_valid_answer_preserved(self):
        weights = self.make_weights(intercept=0.0)
        router = PrefillProbeRouter(weights, threshold_high=0.7, threshold_low=0.3)
        system = EpistemicSteeringSystem(router)
        result = SteeringResult("q1", "mmlu", "cot", 0.5)
        finalized = system.finalize_answer(result, "42")
        assert finalized.final_answer == "42"
        assert finalized.abstained is False

    def test_abstain_route_always_abstains(self):
        weights = self.make_weights(intercept=-10.0)
        router = PrefillProbeRouter(weights, threshold_high=0.7, threshold_low=0.3)
        system = EpistemicSteeringSystem(router)
        result = SteeringResult("q1", "mmlu", "abstain", 0.1)
        finalized = system.finalize_answer(result, "anything")
        assert finalized.final_answer == "I don't know"
        assert finalized.abstained is True


class TestFormatResultsScript:
    def test_format_results_produces_correct_output(self, tmp_path):
        import json
        input_data = [
            {
                "question_id": "q1",
                "dataset": "mmlu",
                "route": "direct",
                "prefill_confidence": 0.8,
                "final_answer": "A",
                "abstained": False,
                "tokens_used": 10,
            },
            {
                "question_id": "q2",
                "dataset": "gsm8k",
                "route": "abstain",
                "prefill_confidence": 0.2,
                "final_answer": "I don't know",
                "abstained": True,
                "tokens_used": 5,
            },
        ]
        input_path = tmp_path / "raw.json"
        output_path = tmp_path / "formatted.json"
        with open(input_path, 'w') as f:
            json.dump(input_data, f)

        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))
        from format_results import format_results
        format_results(str(input_path), str(output_path))

        with open(output_path) as f:
            formatted = json.load(f)

        assert len(formatted) == 2
        assert formatted[0]["question_id"] == "q1"
        assert formatted[0]["predicted_answer"] == "A"
        assert formatted[0]["abstained"] is False
        assert formatted[0]["route_taken"] == "direct"
        assert formatted[1]["predicted_answer"] == "I don't know"
        assert formatted[1]["abstained"] is True
