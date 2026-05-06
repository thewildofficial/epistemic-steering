"""Demonstrate epistemic steering system on sample questions."""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from steering import (
    EpistemicSteeringSystem,
    GenerationTimeMonitor,
    PrefillProbeRouter,
)


def create_mock_weights(hidden_dim: int = 2560, bias: float = 0.0) -> dict:
    """Create mock probe weights for demonstration."""
    rng = np.random.default_rng(42)
    coef = rng.normal(0, 0.01, size=hidden_dim)
    return {
        'coef': coef,
        'intercept': bias,
        'layer': 30,
        'token_position': 'last',
    }


def load_sample_questions(path: str, n: int = 10):
    """Load first N questions from probe_extract_results.jsonl."""
    questions = []
    with open(path, 'r') as f:
        for line in f:
            if len(questions) >= n:
                break
            record = json.loads(line)
            questions.append(record)
    return questions


def load_activation(question_id: str, activations_dir: str, layer: int = 30):
    """Load pre-computed activation for a question."""
    activations_dir = Path(activations_dir)
    pattern = f"{question_id}__layer_{layer}.npy"
    candidates = list(activations_dir.glob(pattern))
    if not candidates:
        return None
    return np.load(candidates[0])


def main():
    project_root = Path(__file__).parent.parent
    data_dir = project_root / 'data'
    activations_dir = data_dir / 'activations'
    results_path = data_dir / 'probe_extract_results.jsonl'

    if not results_path.exists():
        print(f"Results file not found: {results_path}")
        sys.exit(1)

    questions = load_sample_questions(str(results_path), n=10)

    prefill_weights = create_mock_weights(hidden_dim=2560, bias=0.0)
    gen_time_weights = create_mock_weights(hidden_dim=2560, bias=-0.5)

    router = PrefillProbeRouter(
        probe_weights=prefill_weights,
        threshold_high=0.7,
        threshold_low=0.3,
    )
    monitor = GenerationTimeMonitor(
        gen_time_probe_weights=gen_time_weights,
        abstain_threshold=0.3,
        check_every_n_tokens=5,
    )
    steering = EpistemicSteeringSystem(
        prefill_router=router,
        gen_time_monitor=monitor,
    )

    print("=" * 60)
    print("Epistemic Steering System Demo")
    print("=" * 60)

    results = []
    for record in questions:
        qid = record['question_id']
        dataset = record.get('dataset', 'unknown')
        activation = load_activation(qid, str(activations_dir))

        if activation is None:
            rng = np.random.default_rng(hash(qid) % (2**32))
            activation = rng.normal(0, 1, size=2560).astype(np.float32)

        result = steering.route_question(qid, dataset, activation)
        results.append(result)

        print(f"\nQuestion: {qid}")
        print(f"  Dataset:    {result.dataset}")
        print(f"  Confidence: {result.prefill_confidence:.4f}")
        print(f"  Route:      {result.route}")
        if result.abstained:
            print(f"  Answer:     {result.final_answer}")

    stats = steering.get_statistics(results)
    print("\n" + "=" * 60)
    print("Summary Statistics")
    print("=" * 60)
    print(f"Total questions:     {stats['total']}")
    print(f"Direct answers:      {stats['direct_count']} ({stats['direct_pct']:.1%})")
    print(f"CoT + monitor:       {stats['cot_count']} ({stats['cot_pct']:.1%})")
    print(f"Abstained:           {stats['abstain_count']} ({stats['abstain_pct']:.1%})")
    print(f"Mean prefill conf:   {stats['mean_prefill_confidence']:.4f}")


if __name__ == "__main__":
    main()
