"""Baseline: randomly route between direct / CoT / abstain for held-out questions.

Runs on Modal T4 GPU. Evaluates accuracy and token usage when routing
is random rather than based on epistemic confidence.

Usage:
    modal run scripts/evaluate_baseline_random.py --num-questions 200

Cost estimate:
    200 questions x ~10 s avg generation ≈ 0.6 hours on T4
    Estimated total: $0.35-$0.40
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

import modal
from modal import App, Image, Volume

app = App("heldout-baseline-random")

volume = Volume.from_name("epistemic-model-cache")
MODEL_DIR = "/vol/model"
RESULTS_DIR = "/vol/results"
TRAIN_RESULTS_PATH = f"{RESULTS_DIR}/probe_extract_results.jsonl"

image = (
    Image.debian_slim()
    .pip_install(
        "torch",
        "transformers",
        "numpy",
        "datasets",
        "tqdm",
    )
)

MAX_NEW_TOKENS_DIRECT = 10
MAX_NEW_TOKENS_COT = 256


def load_training_data(path: str) -> tuple[set[str], set[str]]:
    """Load training questions and build dedup sets."""
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
    """Load held-out questions from MMLU and GSM8K test sets."""
    from datasets import load_dataset

    heldout: list[dict] = []

    print("Loading MMLU test set ...")
    try:
        mmlu_all = load_dataset("cais/mmlu", "all", split="test", trust_remote_code=True)
        for i, ex in enumerate(mmlu_all):
            prompt = format_mmlu_prompt(ex)
            if prompt in train_prompts:
                continue
            qid = f"mmlu_{ex.get('subject', 'unknown')}_{i}"
            if qid in train_ids:
                continue
            heldout.append(
                {
                    "question_id": qid,
                    "dataset": "mmlu",
                    "prompt": prompt,
                    "correct_answer": ex["answer"],
                }
            )
        print(f"  MMLU held-out: {len(heldout)}")
    except Exception as exc:
        print(f"  WARNING: Could not load MMLU: {exc}")

    print("Loading GSM8K test set ...")
    try:
        gsm8k_test = load_dataset("openai/gsm8k", "main", split="test", trust_remote_code=True)
        gsm8k_count = 0
        for i, ex in enumerate(gsm8k_test):
            prompt = format_gsm8k_prompt(ex)
            if prompt in train_prompts:
                continue
            qid = f"gsm8k_{i}"
            if qid in train_ids:
                continue
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

    random.seed(42)
    random.shuffle(heldout)
    heldout = heldout[:num_questions]

    print(f"Total held-out questions selected: {len(heldout)}")
    return heldout


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


def check_correctness(predicted: str | None, correct: str, dataset: str) -> bool:
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
def evaluate_baseline_random(num_questions: int = 200) -> dict:
    """Run baseline evaluation: random routing."""
    import numpy as np
    import torch
    from tqdm import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer

    start_time = time.time()
    rng = random.Random(42)

    print("=" * 60)
    print("BASELINE: RANDOM ROUTING")
    print("=" * 60)

    print("\n── Loading training data for deduplication ──")
    train_ids, train_prompts = load_training_data(TRAIN_RESULTS_PATH)
    print(f"  Training questions: {len(train_ids)}")

    print("\n── Loading held-out questions ──")
    heldout = load_heldout_questions(train_ids, train_prompts, num_questions)

    if not heldout:
        return {"error": "No held-out questions found"}

    print("\n── Loading model ──")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"  Model loaded on {next(model.parameters()).device}")

    print("\n── Generating with random routing ──")
    results = []
    total_tokens = 0

    for q in tqdm(heldout, desc="Random"):
        prompt = q["prompt"]
        dataset = q["dataset"]
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids.to(model.device)
        attention_mask = inputs.attention_mask.to(model.device)

        # Random route
        route = rng.choice(["direct", "cot", "abstain"])

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
                "final_answer": final_answer,
                "abstained": abstained,
                "tokens_used": tokens_used,
                "correct": correct,
                "correct_answer": q["correct_answer"],
                "generated_text": generated_text,
            }
        )

    # Metrics
    total = len(results)
    correct_count = sum(1 for r in results if r["correct"])
    accuracy = float(correct_count) / float(total) if total > 0 else 0.0
    avg_tokens = float(total_tokens) / float(total) if total > 0 else 0.0

    direct_count = sum(1 for r in results if r["route"] == "direct")
    cot_count = sum(1 for r in results if r["route"] == "cot")
    abstain_count = sum(1 for r in results if r["route"] == "abstain")

    mmlu_results = [r for r in results if r["dataset"] == "mmlu"]
    gsm8k_results = [r for r in results if r["dataset"] == "gsm8k"]

    def dataset_stats(res_list):
        if not res_list:
            return {}
        total_ds = len(res_list)
        correct_ds = sum(1 for r in res_list if r["correct"])
        tokens_ds = sum(r["tokens_used"] for r in res_list)
        return {
            "total": total_ds,
            "correct_count": correct_ds,
            "accuracy": float(correct_ds) / float(total_ds) if total_ds > 0 else 0.0,
            "avg_tokens": float(tokens_ds) / float(total_ds) if total_ds > 0 else 0.0,
        }

    elapsed = time.time() - start_time
    cost_usd = elapsed * 0.000164

    summary = {
        "metadata": {
            "type": "baseline_random",
            "num_questions": num_questions,
            "evaluated": total,
            "model": "Qwen3.5-4B",
            "elapsed_seconds": elapsed,
            "estimated_cost_usd": cost_usd,
        },
        "routing": {
            "total": total,
            "direct_count": direct_count,
            "cot_count": cot_count,
            "abstain_count": abstain_count,
            "direct_pct": float(direct_count) / total if total > 0 else 0.0,
            "cot_pct": float(cot_count) / total if total > 0 else 0.0,
            "abstain_pct": float(abstain_count) / total if total > 0 else 0.0,
        },
        "accuracy": {
            "total_correct": correct_count,
            "total_questions": total,
            "accuracy": accuracy,
        },
        "token_usage": {
            "total_tokens": total_tokens,
            "avg_tokens_per_question": avg_tokens,
        },
        "mmlu": dataset_stats(mmlu_results),
        "gsm8k": dataset_stats(gsm8k_results),
        "results": results,
    }

    output_dir = Path(RESULTS_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "baseline_random_results.json"
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2, default=float)

    print("\n" + "=" * 60)
    print("BASELINE RANDOM COMPLETE")
    print("=" * 60)
    print(f"Accuracy:     {accuracy:.4f}")
    print(f"Avg tokens:   {avg_tokens:.1f}")
    print(f"Direct:       {direct_count} ({direct_count/total:.1%})")
    print(f"CoT:          {cot_count} ({cot_count/total:.1%})")
    print(f"Abstain:      {abstain_count} ({abstain_count/total:.1%})")
    print(f"GPU time:     {elapsed / 60:.1f} min")
    print(f"Cost:         ${cost_usd:.2f}")
    print(f"Saved:        {output_path}")
    print("=" * 60)

    return summary


@app.local_entrypoint()
def main(num_questions: int = 200):
    evaluate_baseline_random.remote(num_questions)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-questions", type=int, default=200)
    args = parser.parse_args()
    print(
        "This script runs on Modal GPU.\n"
        f"  modal run scripts/evaluate_baseline_random.py --num-questions {args.num_questions}"
    )
