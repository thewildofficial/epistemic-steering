from __future__ import annotations

import json
import sys
from pathlib import Path

import importlib.util

spec = importlib.util.spec_from_file_location(
    "re_extract_benchmarks",
    str(Path(__file__).with_name("re_extract_benchmarks.py")),
)
reb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reb)

strip_thinking_blocks = reb.strip_thinking_blocks
check_correctness_humaneval = reb.check_correctness_humaneval
check_correctness_triviaqa = reb.check_correctness_triviaqa

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "benchmark_activations"

HUMANEVAL_RESULTS = DATA_DIR / "humaneval_results.jsonl"
TRIVIAQA_RESULTS = DATA_DIR / "triviaqa_results.jsonl"


def load_jsonl(path: Path) -> list[dict]:
    results = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def save_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r, default=float) + "\n")


def recheck_humaneval() -> tuple[int, int]:
    from datasets import load_dataset

    print("=" * 60)
    print("HumanEval Re-check")
    print("=" * 60)

    results = load_jsonl(HUMANEVAL_RESULTS)
    print(f"Loaded {len(results)} existing HumanEval results")

    ds = load_dataset("openai_humaneval", split="test[:100]")
    dataset_by_task_id = {item["task_id"]: item for item in ds}
    print(f"Loaded {len(dataset_by_task_id)} HumanEval dataset entries")

    n_changed = 0
    n_correct = 0
    for r in results:
        qid = r["question_id"]
        task_id = qid.replace("_", "/")
        ds_entry = dataset_by_task_id.get(task_id)
        if ds_entry is None:
            print(f"  WARNING: no dataset entry for {qid} (task_id={task_id})")
            continue

        prompt = ds_entry["prompt"]
        generated = r["generated_text"]
        correct_answer = r["correct_answer"]

        new_correct = check_correctness_humaneval(prompt, generated, correct_answer)

        if new_correct != r.get("correct", False):
            n_changed += 1
            print(f"  {qid}: {r.get('correct', False)} -> {new_correct}")

        r["correct"] = new_correct
        if new_correct:
            n_correct += 1

    save_jsonl(HUMANEVAL_RESULTS, results)
    print(f"\nHumanEval: {n_correct}/{len(results)} correct ({n_correct / len(results):.1%})")
    print(f"  Changed: {n_changed} verdicts")
    return n_correct, len(results)


def recheck_triviaqa() -> tuple[int, int]:
    from datasets import load_dataset

    print("\n" + "=" * 60)
    print("TriviaQA Re-check")
    print("=" * 60)

    results = load_jsonl(TRIVIAQA_RESULTS)
    print(f"Loaded {len(results)} existing TriviaQA results")

    ds = load_dataset("trivia_qa", "rc.nocontext", split="validation[:100]")
    dataset_by_index = {i: item for i, item in enumerate(ds)}
    print(f"Loaded {len(dataset_by_index)} TriviaQA dataset entries")

    n_changed = 0
    n_correct = 0
    for r in results:
        qid = r["question_id"]
        idx = int(qid.split("_")[-1])
        ds_entry = dataset_by_index.get(idx)
        if ds_entry is None:
            print(f"  WARNING: no dataset entry for {qid}")
            continue

        generated = r["generated_text"]
        aliases = ds_entry["answer"]["value"]
        if not isinstance(aliases, list):
            aliases = [str(aliases)]

        new_correct = False
        for alias in aliases:
            if check_correctness_triviaqa(generated, str(alias)):
                new_correct = True
                break

        if new_correct != r.get("correct", False):
            n_changed += 1
            print(f"  {qid}: {r.get('correct', False)} -> {new_correct}  (aliases: {aliases})")

        r["correct"] = new_correct
        if new_correct:
            n_correct += 1

    save_jsonl(TRIVIAQA_RESULTS, results)
    print(f"\nTriviaQA: {n_correct}/{len(results)} correct ({n_correct / len(results):.1%})")
    print(f"  Changed: {n_changed} verdicts")
    return n_correct, len(results)


def main():
    he_correct, he_total = recheck_humaneval()
    tq_correct, tq_total = recheck_triviaqa()

    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"  HumanEval: {he_correct}/{he_total} = {he_correct / he_total:.1%}")
    print(f"  TriviaQA:  {tq_correct}/{tq_total} = {tq_correct / tq_total:.1%}")

    if he_correct / he_total < 0.15:
        print("\nWARNING: HumanEval accuracy is still below 15%!")
        print("Investigating what's going wrong...")
        results = load_jsonl(HUMANEVAL_RESULTS)
        failures = [r for r in results if not r.get("correct")]
        print(f"  First 5 failures:")
        for r in failures[:5]:
            print(f"    {r['question_id']}: generated length={len(r['generated_text'])}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
