"""Evaluate epistemic steering on held-out questions with batch processing.

Runs on Modal T4 GPU. Uses NEW questions not in the 656-question training set.
Batch processing reduces runtime from 2+ hours to ~25 minutes for 200 questions.

Usage:
    modal run --detach scripts/evaluate_heldout.py --num-questions 200

Cost estimate:
    200 questions x ~7 s generation each (batch_size=4) ≈ 25 min on T4
    T4 cost: $0.000164/sec ≈ $0.59/hr
    Estimated total: $0.25-$0.50
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
MODEL_DIR = "/vol/models/Qwen_Qwen3.5-4B"
RESULTS_DIR = "/vol/results"
ACTIVATIONS_DIR = "/vol/results/activations"
TRAIN_RESULTS_PATH = f"{RESULTS_DIR}/probe_extract_results.jsonl"

image = (
    Image.debian_slim()
    .pip_install(
        "torch",
        "transformers",
        "numpy",
        "tqdm",
        "scikit-learn",
        "scipy",
        "pandas",
        "accelerate",
    )
)

LAYER = 30
HIDDEN_DIM = 2560
PREFILL_HIGH = 0.7
PREFILL_LOW = 0.3
MAX_NEW_TOKENS_DIRECT = 10
MAX_NEW_TOKENS_COT = 256
BATCH_SIZE = 4





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
    """Load held-out questions from pre-saved file on Modal volume."""
    import json
    
    heldout_path = "/vol/results/heldout_all_subjects.jsonl"
    heldout = []
    
    with open(heldout_path) as f:
        for line in f:
            q = json.loads(line)
            if q["question_id"] not in train_ids:
                heldout.append(q)
    
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
    if len(text) == 1 and text in "ABCD":
        return text
    matches = re.findall(r"\b([A-D])\b", text)
    if matches:
        return matches[-1]
    return None


def extract_answer_gsm8k(text: str) -> str | None:
    """Extract final number from generated text."""
    if not text:
        return None
    text = text.strip()
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
    if numbers:
        return numbers[-1]
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
        try:
            return float(predicted) == float(correct)
        except ValueError:
            return predicted == correct
    return predicted == correct



@app.function(
    gpu="T4",
    volumes={"/vol": volume},
    image=image,
    timeout=7200,
)
def evaluate_heldout(num_questions: int = 200) -> dict:
    """Run full held-out evaluation with batch processing."""
    import numpy as np
    import torch
    from scipy.special import expit
    from tqdm import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sklearn.metrics import roc_auc_score, confusion_matrix

    start_time = time.time()

    # 1. Load training data for leakage prevention
    print("=" * 60)
    print("HELD-OUT EVALUATION (BATCHED)")
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
    tokenizer.padding_side = "left"

    print(f"  Model loaded on {next(model.parameters()).device}")

    # 4. Compute training-free probe weights
    print("\n── Computing training-free probe weights ──")
    probe_weights = compute_training_free_probe_weights(
        TRAIN_RESULTS_PATH, ACTIVATIONS_DIR, LAYER
    )

    print("\n── Evaluating held-out questions ──")
    results = []
    total_tokens = 0

    output_dir = Path(RESULTS_DIR) / "heldout_eval"
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "heldout_results.jsonl"
    with open(jsonl_path, "w") as _:
        pass

    num_batches = (len(heldout) + BATCH_SIZE - 1) // BATCH_SIZE
    batch_iter = tqdm(range(0, len(heldout), BATCH_SIZE), total=num_batches, desc="Batches")

    for batch_start in batch_iter:
        batch = heldout[batch_start:batch_start + BATCH_SIZE]

        prompts = []
        for q in batch:
            if q["dataset"] == "gsm8k":
                msgs = [{"role": "user", "content": q["prompt"]}]
                chat_prompt = tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True
                )
                prompts.append(chat_prompt)
            else:
                prompts.append(q["prompt"])

        inputs = tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True
        ).to(model.device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states[LAYER][:, -1, :].cpu().numpy()

        batch_items = []
        for i, q in enumerate(batch):
            hs = hidden_states[i]
            w = probe_weights.get(q["dataset"])
            if w is None:
                w = probe_weights.get("mmlu", {
                    "coef": np.zeros(HIDDEN_DIM, dtype=np.float32),
                    "intercept": 0.0,
                })

            direction = w["coef"]
            intercept = w["intercept"]
            score = float(expit(np.dot(hs, direction) + intercept))

            if score >= PREFILL_HIGH:
                route = "direct"
            elif score <= PREFILL_LOW:
                route = "abstain"
            else:
                route = "cot"

            batch_items.append((q, score, route, i))

        direct_items = [(q, s, i) for q, s, r, i in batch_items if r == "direct"]
        cot_items = [(q, s, i) for q, s, r, i in batch_items if r == "cot"]
        abstain_items = [(q, s, i) for q, s, r, i in batch_items if r == "abstain"]

        for q, score, _ in abstain_items:
            result = {
                "question_id": q["question_id"],
                "dataset": q["dataset"],
                "probe_score": score,
                "route": "abstain",
                "generated_text": "",
                "model_answer": "I don't know",
                "correct_answer": q["correct_answer"],
                "correct": False,
                "tokens_used": 0,
            }
            results.append(result)
            with open(jsonl_path, "a") as f:
                f.write(json.dumps(result, default=float) + "\n")

        if direct_items:
            indices = [i for _, _, i in direct_items]
            batch_input_ids = inputs.input_ids[indices]
            batch_attention_mask = inputs.attention_mask[indices]

            with torch.no_grad():
                gen_ids = model.generate(
                    input_ids=batch_input_ids,
                    attention_mask=batch_attention_mask,
                    max_new_tokens=MAX_NEW_TOKENS_DIRECT,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )

            input_len = batch_input_ids.shape[1]
            for idx, (q, score, _) in enumerate(direct_items):
                generated_ids = gen_ids[idx, input_len:]
                generated_text = tokenizer.decode(
                    generated_ids, skip_special_tokens=True
                )
                tokens_used = int((generated_ids != tokenizer.pad_token_id).sum())

                if q["dataset"] == "mmlu":
                    model_answer = extract_answer_mmlu(generated_text)
                else:
                    model_answer = extract_answer_gsm8k(generated_text)

                if model_answer is None:
                    model_answer = "I don't know"

                correct = check_correctness(
                    model_answer, q["correct_answer"], q["dataset"]
                )
                total_tokens += tokens_used

                result = {
                    "question_id": q["question_id"],
                    "dataset": q["dataset"],
                    "probe_score": score,
                    "route": "direct",
                    "generated_text": generated_text,
                    "model_answer": model_answer,
                    "correct_answer": q["correct_answer"],
                    "correct": correct,
                    "tokens_used": tokens_used,
                }
                results.append(result)
                with open(jsonl_path, "a") as f:
                    f.write(json.dumps(result, default=float) + "\n")

        if cot_items:
            indices = [i for _, _, i in cot_items]
            batch_input_ids = inputs.input_ids[indices]
            batch_attention_mask = inputs.attention_mask[indices]

            with torch.no_grad():
                gen_ids = model.generate(
                    input_ids=batch_input_ids,
                    attention_mask=batch_attention_mask,
                    max_new_tokens=MAX_NEW_TOKENS_COT,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )

            input_len = batch_input_ids.shape[1]
            for idx, (q, score, _) in enumerate(cot_items):
                generated_ids = gen_ids[idx, input_len:]
                generated_text = tokenizer.decode(
                    generated_ids, skip_special_tokens=True
                )
                tokens_used = int((generated_ids != tokenizer.pad_token_id).sum())

                if q["dataset"] == "mmlu":
                    model_answer = extract_answer_mmlu(generated_text)
                else:
                    model_answer = extract_answer_gsm8k(generated_text)

                if model_answer is None:
                    model_answer = "I don't know"

                correct = check_correctness(
                    model_answer, q["correct_answer"], q["dataset"]
                )
                total_tokens += tokens_used

                result = {
                    "question_id": q["question_id"],
                    "dataset": q["dataset"],
                    "probe_score": score,
                    "route": "cot",
                    "generated_text": generated_text,
                    "model_answer": model_answer,
                    "correct_answer": q["correct_answer"],
                    "correct": correct,
                    "tokens_used": tokens_used,
                }
                results.append(result)
                with open(jsonl_path, "a") as f:
                    f.write(json.dumps(result, default=float) + "\n")

    print("\n── Computing metrics ──")

    confidences = np.array([r["probe_score"] for r in results])
    labels = np.array([r["correct"] for r in results])

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
        incorrect = ~labels_arr
        caught = np.sum(incorrect & (scores < threshold))
        prevention_rate = float(caught / np.sum(incorrect)) if np.sum(incorrect) > 0 else 0.0
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
        tokens_per_correct = tokens_per_question
        if cot_tokens > 0:
            savings_vs_cot = float(cot_tokens - routed_tokens) / float(cot_tokens)
        else:
            savings_vs_cot = 0.0
        return {
            "tokens_per_question": tokens_per_question,
            "tokens_per_correct": tokens_per_correct,
            "savings_vs_cot": savings_vs_cot,
        }

    probe_metrics = _compute_all_metrics(confidences, labels, threshold=0.5)

    direct_correct = sum(
        1 for r in results if r["route"] == "direct" and r["correct"]
    )
    cot_correct = sum(
        1 for r in results if r["route"] == "cot" and r["correct"]
    )
    abstentions = sum(1 for r in results if r["route"] == "abstain")
    total = len(results)

    sel_acc = _selective_accuracy(direct_correct, cot_correct, abstentions, total)

    direct_tokens = sum(
        r["tokens_used"] for r in results if r["route"] == "direct"
    )
    cot_tokens_used = sum(
        r["tokens_used"] for r in results if r["route"] == "cot"
    )
    routed_tokens = direct_tokens + cot_tokens_used

    avg_cot_tokens = (
        float(cot_tokens_used) / sum(1 for r in results if r["route"] == "cot")
        if any(r["route"] == "cot" for r in results)
        else 100.0
    )
    estimated_always_cot_tokens = avg_cot_tokens * total

    tok_eff = _token_efficiency(
        direct_tokens, estimated_always_cot_tokens, routed_tokens, total
    )

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
            "batch_size": BATCH_SIZE,
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

    summary_path = output_dir / "heldout_summary.json"
    with open(summary_path, "w") as f:
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
    print(f"Results JSONL:       {jsonl_path}")
    print(f"Summary JSON:        {summary_path}")
    print("=" * 60)

    return summary


@app.local_entrypoint()
def main(num_questions: int = 200):
    evaluate_heldout.remote(num_questions)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-questions", type=int, default=200)
    args = parser.parse_args()
    print(
        "This script is designed to run on Modal GPU.\n"
        "To execute (detached):\n"
        f"  modal run --detach scripts/evaluate_heldout.py --num-questions {args.num_questions}\n"
        "Or attached:\n"
        f"  modal run scripts/evaluate_heldout.py --num-questions {args.num_questions}\n"
    )
