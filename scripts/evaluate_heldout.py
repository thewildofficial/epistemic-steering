"""Evaluate epistemic steering on held-out questions.

Runs on Modal T4 GPU. Uses NEW questions not in the 656-question training set.
Loads steering system, routes each question, computes all metrics.

Usage:
    uv run python scripts/evaluate_heldout.py --num-questions 200

Cost estimate:
    200 questions x ~15-30 s generation each ≈ 0.8-1.7 hours on T4
    T4 cost: $0.000164/sec ≈ $0.59/hr
    Estimated total: $0.50-$1.00
    Budget ceiling: $10
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path

import modal
from modal import App, Image, Volume

app = App("heldout-evaluation")

volume = Volume.from_name("epistemic-model-cache")
MODEL_DIR = "/vol/model"
RESULTS_DIR = "/vol/results"
ACTIVATIONS_DIR = "/vol/results/activations"
TRAIN_RESULTS_PATH = f"{RESULTS_DIR}/probe_extract_results.jsonl"

image = (
    Image.debian_slim()
    .pip_install(
        "torch",
        "transformers",
        "numpy",
        "datasets",
        "tqdm",
        "scikit-learn",
        "scipy",
        "pandas",
    )
)

LAYER = 30
HIDDEN_DIM = 2560
PREFILL_HIGH = 0.7
PREFILL_LOW = 0.3
MAX_NEW_TOKENS_DIRECT = 10
MAX_NEW_TOKENS_COT = 256


# ── Helper functions (run inside Modal container) ───────────────────────────


def load_training_data(path: str) -> tuple[set[str], set[str]]:
    """Load training questions and build dedup sets.

    Returns:
        (train_ids, train_prompts)
    """
    train_ids: set[str] = set()
    train_prompts: set[str] = set()

    with open(path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            q = json.loads(line)
            train_ids.add(q["question_id"])
            train_prompts.add(q["prompt"])

    return train_ids, train_prompts


def format_mmlu_prompt(example: dict) -> str:
    """Format an MMLU example into the prompt used in training."""
    choices = example.get("choices", [])
    choice_text = ""
    for i, choice in enumerate(choices):
        letter = chr(ord("A") + i)
        choice_text += f"{letter}) {choice}\n"

    prompt = (
        "Answer the following multiple choice question. "
        "Respond with ONLY the letter (A, B, C, or D). "
        "Do not explain your reasoning.\n\n"
        f"Question: {example['question']}\n"
        f"{choice_text}\n"
        "Answer:"
    )
    return prompt


def format_gsm8k_prompt(example: dict) -> str:
    """Format a GSM8K example into the prompt used in training."""
    prompt = (
        "Solve this math problem. Show your work step by step, "
        "then give the final answer as a number.\n\n"
        f"Question: {example['question']}\n\n"
        "Answer:"
    )
    return prompt


def load_heldout_questions(
    train_ids: set[str],
    train_prompts: set[str],
    num_questions: int,
) -> list[dict]:
    """Load held-out questions from MMLU and GSM8K test sets.

    Filters out any questions that appear in the training set
    (matched by prompt content to guarantee no leakage).
    """
    from datasets import load_dataset

    heldout: list[dict] = []

    # ── MMLU ────────────────────────────────────────────────────────────────
    print("Loading MMLU test set ...")
    try:
        mmlu_all = load_dataset("cais/mmlu", "all", split="test", trust_remote_code=True)
        for i, ex in enumerate(mmlu_all):
            prompt = format_mmlu_prompt(ex)
            # Skip if this exact prompt was in training data
            if prompt in train_prompts:
                continue
            qid = f"mmlu_{ex.get('subject', 'unknown')}_{i}"
            # Also skip if ID somehow matches
            if qid in train_ids:
                continue
            heldout.append(
                {
                    "question_id": qid,
                    "dataset": "mmlu",
                    "subject": ex.get("subject", "unknown"),
                    "prompt": prompt,
                    "correct_answer": ex["answer"],
                }
            )
        print(f"  MMLU held-out: {len(heldout)}")
    except Exception as exc:
        print(f"  WARNING: Could not load MMLU: {exc}")

    # ── GSM8K ───────────────────────────────────────────────────────────────
    print("Loading GSM8K test set ...")
    try:
        gsm8k_test = load_dataset(
            "openai/gsm8k", "main", split="test", trust_remote_code=True
        )
        gsm8k_count = 0
        for i, ex in enumerate(gsm8k_test):
            prompt = format_gsm8k_prompt(ex)
            if prompt in train_prompts:
                continue
            qid = f"gsm8k_{i}"
            if qid in train_ids:
                continue
            # GSM8K answer is after "#### " in the original
            answer_text = ex["answer"].split("####")[-1].strip()
            heldout.append(
                {
                    "question_id": qid,
                    "dataset": "gsm8k",
                    "prompt": prompt,
                    "correct_answer": answer_text,
                }
            )
            gsm8k_count += 1
        print(f"  GSM8K held-out: {gsm8k_count}")
    except Exception as exc:
        print(f"  WARNING: Could not load GSM8K: {exc}")

    # Shuffle and sample
    random.seed(42)
    random.shuffle(heldout)
    heldout = heldout[:num_questions]

    print(f"Total held-out questions selected: {len(heldout)}")
    mmlu_n = sum(1 for q in heldout if q["dataset"] == "mmlu")
    gsm8k_n = sum(1 for q in heldout if q["dataset"] == "gsm8k")
    print(f"  MMLU: {mmlu_n}, GSM8K: {gsm8k_n}")

    return heldout


def compute_training_free_probe_weights(
    train_results_path: str,
    activations_dir: str,
    layer: int = 30,
) -> dict:
    """Compute training-free probe weights from in-sample data.

    Uses per-dataset mean-difference directions (MMLU and GSM8K)
    to avoid dataset-specific patterns diluting separation.

    Returns:
        dict with keys 'mmlu' and 'gsm8k', each containing
        {'coef': np.ndarray, 'intercept': float}
    """
    import numpy as np
    from scipy.special import expit

    # Load training records
    records = []
    with open(train_results_path, "r") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    activations_dir = Path(activations_dir)

    acts_by_dataset: dict[str, list[tuple[np.ndarray, bool]]] = {
        "mmlu": [],
        "gsm8k": [],
    }

    for rec in records:
        qid = rec["question_id"]
        dataset = rec["dataset"]
        correct = rec["correct"]

        if dataset not in acts_by_dataset:
            continue

        npy_path = activations_dir / f"{qid}__layer_{layer}.npy"
        if not npy_path.exists():
            npy_path = activations_dir / f"q{qid}_layer_{layer}.npy"
        if not npy_path.exists():
            continue

        try:
            act = np.load(npy_path).ravel()
            acts_by_dataset[dataset].append((act, correct))
        except Exception:
            continue

    probe_weights = {}

    for dataset, items in acts_by_dataset.items():
        if len(items) < 10:
            print(
                f"  WARNING: Only {len(items)} activations for {dataset}, "
                "using zero probe"
            )
            probe_weights[dataset] = {
                "coef": np.zeros(HIDDEN_DIM, dtype=np.float32),
                "intercept": 0.0,
            }
            continue

        acts = np.array([item[0] for item in items])
        labels = np.array([item[1] for item in items], dtype=bool)

        correct_acts = acts[labels]
        incorrect_acts = acts[~labels]

        if len(correct_acts) == 0 or len(incorrect_acts) == 0:
            print(
                f"  WARNING: No class balance for {dataset}, using zero probe"
            )
            probe_weights[dataset] = {
                "coef": np.zeros(HIDDEN_DIM, dtype=np.float32),
                "intercept": 0.0,
            }
            continue

        direction = correct_acts.mean(axis=0) - incorrect_acts.mean(axis=0)
        direction_norm = np.linalg.norm(direction)
        if direction_norm > 0:
            direction = direction / direction_norm

        midpoint = (correct_acts.mean(axis=0) + incorrect_acts.mean(axis=0)) / 2.0
        intercept = float(-np.dot(direction, midpoint))

        # Quick validation
        projections = np.dot(acts - midpoint, direction)
        scores = expit(projections)
        from sklearn.metrics import roc_auc_score
        try:
            auroc = float(roc_auc_score(labels, scores))
        except Exception:
            auroc = float("nan")

        print(
            f"  {dataset}: n={len(acts)} correct={labels.sum()} "
            f"incorrect={(~labels).sum()} AUROC={auroc:.3f}"
        )

        probe_weights[dataset] = {
            "coef": direction.astype(np.float32),
            "intercept": intercept,
        }

    return probe_weights


def extract_answer_mmlu(text: str) -> str | None:
    """Extract A/B/C/D from generated text."""
    if not text:
        return None
    text = text.strip()
    # For direct answers, usually just one letter at start
    if len(text) == 1 and text in "ABCD":
        return text
    # Search for any A/B/C/D
    matches = re.findall(r"\b([A-D])\b", text)
    if matches:
        return matches[-1]  # last match (for CoT, answer is usually at end)
    return None


def extract_answer_gsm8k(text: str) -> str | None:
    """Extract final number from generated text."""
    if not text:
        return None
    text = text.strip()
    # Look for numbers (including decimals and negatives)
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
    if numbers:
        return numbers[-1]  # last number is usually the final answer
    return None


def check_correctness(
    predicted: str | None, correct: str, dataset: str
) -> bool:
    """Check if predicted answer matches correct answer."""
    if predicted is None:
        return False
    predicted = predicted.strip()
    correct = correct.strip()

    if dataset == "mmlu":
        return predicted.upper() == correct.upper()
    elif dataset == "gsm8k":
        # Compare as numbers to handle "42" vs "42.0"
        try:
            return float(predicted) == float(correct)
        except ValueError:
            return predicted == correct
    return predicted == correct


# ── Modal GPU function ──────────────────────────────────────────────────────


@app.function(
    gpu="T4",
    volumes={"/vol": volume},
    image=image,
    timeout=7200,
)
def evaluate_heldout(num_questions: int = 200) -> dict:
    """Run full held-out evaluation."""
    import numpy as np
    import torch
    from tqdm import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer

    start_time = time.time()

    # 1. Load training data for leakage prevention
    print("=" * 60)
    print("HELD-OUT EVALUATION")
    print("=" * 60)
    print("\n── Loading training data for deduplication ──")
    train_ids, train_prompts = load_training_data(TRAIN_RESULTS_PATH)
    print(f"  Training questions: {len(train_ids)}")

    # 2. Load held-out questions
    print("\n── Loading held-out questions ──")
    heldout = load_heldout_questions(train_ids, train_prompts, num_questions)

    if not heldout:
        print("ERROR: No held-out questions found.")
        return {"error": "No held-out questions found"}

    # 3. Load model
    print("\n── Loading model ──")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"  Model loaded on {next(model.parameters()).device}")

    # 4. Compute training-free probe weights
    print("\n── Computing training-free probe weights ──")
    probe_weights = compute_training_free_probe_weights(
        TRAIN_RESULTS_PATH, ACTIVATIONS_DIR, LAYER
    )

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.steering import EpistemicSteeringSystem, PrefillProbeRouter
    from src.evaluate import compute_all_metrics, selective_accuracy, token_efficiency

    # 5. Evaluate each question
    print("\n── Evaluating held-out questions ──")
    results = []
    total_tokens = 0

    for q in tqdm(heldout, desc="Evaluating"):
        prompt = q["prompt"]
        dataset = q["dataset"]
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids.to(model.device)
        attention_mask = inputs.attention_mask.to(model.device)

        # Extract prefill activation at layer 30
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
        hidden = outputs.hidden_states[LAYER][0, -1, :].cpu().numpy()

        w = probe_weights.get(dataset, probe_weights.get("mmlu"))
        router = PrefillProbeRouter(
            probe_weights=w,
            threshold_high=PREFILL_HIGH,
            threshold_low=PREFILL_LOW,
        )
        system = EpistemicSteeringSystem(prefill_router=router)
        result = system.route_question(q["question_id"], dataset, hidden)
        confidence = result.prefill_confidence
        route = result.route

        # Generate based on route
        if route == "abstain":
            final_answer = "I don't know"
            abstained = True
            tokens_used = 0
            generated_text = ""
        elif route == "direct":
            with torch.no_grad():
                output_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=MAX_NEW_TOKENS_DIRECT,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            generated_text = tokenizer.decode(
                output_ids[0][input_ids.shape[1] :],
                skip_special_tokens=True,
            )
            tokens_used = int(output_ids.shape[1] - input_ids.shape[1])
            if dataset == "mmlu":
                final_answer = extract_answer_mmlu(generated_text)
            else:
                final_answer = extract_answer_gsm8k(generated_text)
            abstained = final_answer is None
            if abstained:
                final_answer = "I don't know"
        else:  # cot
            with torch.no_grad():
                output_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=MAX_NEW_TOKENS_COT,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            generated_text = tokenizer.decode(
                output_ids[0][input_ids.shape[1] :],
                skip_special_tokens=True,
            )
            tokens_used = int(output_ids.shape[1] - input_ids.shape[1])
            if dataset == "mmlu":
                final_answer = extract_answer_mmlu(generated_text)
            else:
                final_answer = extract_answer_gsm8k(generated_text)
            abstained = final_answer is None
            if abstained:
                final_answer = "I don't know"

        total_tokens += tokens_used

        correct = check_correctness(final_answer, q["correct_answer"], dataset)

        results.append(
            {
                "question_id": q["question_id"],
                "dataset": dataset,
                "route": route,
                "prefill_confidence": float(confidence),
                "final_answer": final_answer,
                "abstained": abstained,
                "tokens_used": tokens_used,
                "correct": correct,
                "correct_answer": q["correct_answer"],
                "generated_text": generated_text,
            }
        )

    # 6. Compute metrics
    print("\n── Computing metrics ──")

    # Build arrays for probe metrics
    confidences = np.array([r["prefill_confidence"] for r in results])
    labels = np.array([r["correct"] for r in results])

    # Inline metrics to avoid container src path issues
    from sklearn.metrics import roc_auc_score, confusion_matrix

    def _compute_all_metrics(scores, labels_arr, threshold=0.5):
        labels_arr = np.asarray(labels_arr)
        auroc = float(roc_auc_score(labels_arr, scores))
        cm = confusion_matrix(labels_arr, scores >= threshold)
        tp, fp, tn, fn = cm[1, 1], cm[0, 1], cm[0, 0], cm[1, 0]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        # Prevention rate: fraction of incorrect caught
        incorrect = ~labels_arr
        caught = np.sum(incorrect & (scores < threshold))
        prevention_rate = float(caught / np.sum(incorrect)) if np.sum(incorrect) > 0 else 0.0
        # Unnecessary block rate: fraction of correct blocked
        correct = labels_arr
        blocked = np.sum(correct & (scores < threshold))
        unnecessary_block_rate = (
            float(blocked / np.sum(correct)) if np.sum(correct) > 0 else 0.0
        )
        return {
            "auroc": auroc,
            "threshold": threshold,
            "confusion_matrix": {"TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn)},
            "prevention_rate": prevention_rate,
            "unnecessary_block_rate": unnecessary_block_rate,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
        }

    def _selective_accuracy(direct_correct, cot_correct, abstentions, total):
        if total == 0:
            return 0.0
        return float(direct_correct + cot_correct) / float(total)

    def _token_efficiency(direct_tokens, cot_tokens, routed_tokens, total):
        if total == 0:
            return {
                "tokens_per_question": 0.0,
                "tokens_per_correct": 0.0,
                "savings_vs_cot": 0.0,
            }
        tokens_per_question = float(routed_tokens) / float(total)
        tokens_per_correct = tokens_per_question  # placeholder
        if cot_tokens > 0:
            savings_vs_cot = float(cot_tokens - routed_tokens) / float(cot_tokens)
        else:
            savings_vs_cot = 0.0
        return {
            "tokens_per_question": tokens_per_question,
            "tokens_per_correct": tokens_per_correct,
            "savings_vs_cot": savings_vs_cot,
        }

    probe_metrics = compute_all_metrics(confidences, labels, threshold=0.5)

    direct_correct = sum(
        1 for r in results if r["route"] == "direct" and r["correct"]
    )
    cot_correct = sum(
        1 for r in results if r["route"] == "cot" and r["correct"]
    )
    abstentions = sum(1 for r in results if r["route"] == "abstain")
    total = len(results)

    sel_acc = selective_accuracy(direct_correct, cot_correct, abstentions, total)

    direct_tokens = sum(
        r["tokens_used"] for r in results if r["route"] == "direct"
    )
    cot_tokens_used = sum(
        r["tokens_used"] for r in results if r["route"] == "cot"
    )
    routed_tokens = direct_tokens + cot_tokens_used

    # For savings_vs_cot, estimate what always-CoT would cost
    # (use average CoT tokens per question * total questions)
    avg_cot_tokens = (
        float(cot_tokens_used) / sum(1 for r in results if r["route"] == "cot")
        if any(r["route"] == "cot" for r in results)
        else 100.0
    )
    estimated_always_cot_tokens = avg_cot_tokens * total

    tok_eff = token_efficiency(
        direct_tokens, estimated_always_cot_tokens, routed_tokens, total
    )

    # Per-dataset breakdown
    mmlu_results = [r for r in results if r["dataset"] == "mmlu"]
    gsm8k_results = [r for r in results if r["dataset"] == "gsm8k"]

    def dataset_stats(res_list):
        if not res_list:
            return {}
        total_ds = len(res_list)
        direct_ds = sum(1 for r in res_list if r["route"] == "direct")
        cot_ds = sum(1 for r in res_list if r["route"] == "cot")
        abstain_ds = sum(1 for r in res_list if r["route"] == "abstain")
        correct_ds = sum(1 for r in res_list if r["correct"])
        return {
            "total": total_ds,
            "direct_count": direct_ds,
            "cot_count": cot_ds,
            "abstain_count": abstain_ds,
            "correct_count": correct_ds,
            "accuracy": float(correct_ds) / float(total_ds) if total_ds > 0 else 0.0,
            "direct_pct": float(direct_ds) / float(total_ds) if total_ds > 0 else 0.0,
            "cot_pct": float(cot_ds) / float(total_ds) if total_ds > 0 else 0.0,
            "abstain_pct": float(abstain_ds) / float(total_ds) if total_ds > 0 else 0.0,
        }

    elapsed = time.time() - start_time
    cost_usd = elapsed * 0.000164

    summary = {
        "metadata": {
            "type": "held_out_evaluation",
            "num_questions": num_questions,
            "evaluated": total,
            "model": "Qwen3.5-4B",
            "probe_layer": LAYER,
            "elapsed_seconds": elapsed,
            "estimated_cost_usd": cost_usd,
            "warning": (
                "These are OUT-OF-SAMPLE results using training-free probe. "
                "Probe was NOT trained on held-out data, but direction was "
                "computed from in-sample activations. Generalization is not guaranteed."
            ),
        },
        "routing": {
            "total": total,
            "direct_count": sum(1 for r in results if r["route"] == "direct"),
            "cot_count": sum(1 for r in results if r["route"] == "cot"),
            "abstain_count": abstentions,
            "direct_pct": (
                float(sum(1 for r in results if r["route"] == "direct")) / total
                if total > 0
                else 0.0
            ),
            "cot_pct": (
                float(sum(1 for r in results if r["route"] == "cot")) / total
                if total > 0
                else 0.0
            ),
            "abstain_pct": (
                float(abstentions) / total if total > 0 else 0.0
            ),
        },
        "accuracy": {
            "selective_accuracy": sel_acc,
            "direct_correct": direct_correct,
            "cot_correct": cot_correct,
            "total_correct": direct_correct + cot_correct,
            "total_questions": total,
        },
        "token_efficiency": tok_eff,
        "probe_metrics": probe_metrics,
        "mmlu": dataset_stats(mmlu_results),
        "gsm8k": dataset_stats(gsm8k_results),
        "results": results,
    }

    # 7. Save results
    output_dir = Path(RESULTS_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "heldout_evaluation_results.json"
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2, default=float)

    print("\n" + "=" * 60)
    print("HELD-OUT EVALUATION COMPLETE")
    print("=" * 60)
    print(f"Questions evaluated: {total}")
    print(f"Selective accuracy:  {sel_acc:.4f}")
    print(f"Probe AUROC:         {probe_metrics['auroc']:.4f}")
    print(f"Prevention rate:     {probe_metrics['prevention_rate']:.4f}")
    print(f"Unnecessary block:   {probe_metrics['unnecessary_block_rate']:.4f}")
    print(f"Tokens per question: {tok_eff['tokens_per_question']:.1f}")
    print(f"Savings vs always-CoT: {tok_eff['savings_vs_cot']:.2%}")
    print(f"GPU time:            {elapsed / 60:.1f} min")
    print(f"Estimated cost:      ${cost_usd:.2f}")
    print(f"Results saved:       {output_path}")
    print("=" * 60)

    return summary


@app.local_entrypoint()
def main(num_questions: int = 200):
    evaluate_heldout.remote(num_questions)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-questions", type=int, default=200)
    args = parser.parse_args()
    # When run locally (not via Modal CLI), print instructions
    print(
        "This script is designed to run on Modal GPU.\n"
        "To execute:\n"
        f"  modal run scripts/evaluate_heldout.py --num-questions {args.num_questions}\n"
        "Or:\n"
        f"  uv run python scripts/evaluate_heldout.py --num-questions {args.num_questions}\n"
        "(if configured with Modal local entrypoint)"
    )
