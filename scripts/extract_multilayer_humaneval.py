"""Extract multi-layer hidden states, MSP, and entropy for HumanEval on Modal GPU.

Adapts patterns from extract_benchmark_activations.py and re_extract_benchmarks.py.
Extracts hidden states at ALL specified layers during a single forward pass,
computes per-sample MSP (mean max softmax) and entropy, saves generated text
for LVU parsing, and tags each sample with an 80/20 train/val split.

Usage (Modal GPU extraction):
    modal run --detach scripts/extract_multilayer_humaneval.py

Usage (local prompt validation):
    python scripts/extract_multilayer_humaneval.py --validate-prompts --n_samples 1
"""

from __future__ import annotations

import argparse
import json
import re
import textwrap
import time
from pathlib import Path

import modal
from modal import App, Image, Volume

app = App("multilayer-humaneval-extraction")

volume = Volume.from_name("epistemic-model-cache")
MODEL_DIR = "/vol/models/Qwen_Qwen3.5-4B"
RESULTS_DIR = "/vol/results"

image = (
    Image.debian_slim()
    .pip_install(
        "torch",
        "transformers",
        "numpy",
        "tqdm",
        "accelerate",
        "datasets",
    )
)

HIDDEN_DIM = 2560
BATCH_SIZE = 2
SEED = 42

DEFAULT_LAYERS = [15, 17, 19, 20, 22, 25]
DEFAULT_N_SAMPLES = 100
DEFAULT_MAX_NEW_TOKENS = 2048

HUMANEVAL_FEWSHOT = """Complete the following Python function.

Example 1:
def has_close_elements(numbers: List[float], threshold: float) -> bool:
    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than
    given threshold.
    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
    False
    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
    True
    \"\"\"
    for i in range(len(numbers)):
        for j in range(i + 1, len(numbers)):
            if abs(numbers[i] - numbers[j]) < threshold:
                return True
    return False

Example 2:
def separate_paren_groups(paren_string: str):
    \"\"\" Input to this function is a string containing multiple groups of nested parentheses. Your goal is to
    separate those group into separate strings and return the list of those strings.
    Separate groups are balanced (each open brace is properly closed) and not nested within each other.
    Ignore any spaces in the input string.
    >>> separate_paren_groups('( ) (( )) (( )( ))')
    ['()', '(())', '(()())']
    \"\"\"
    result = []
    current_string = []
    current_depth = 0
    for c in paren_string:
        if c == '(':
            current_depth += 1
            current_string.append(c)
        elif c == ')':
            current_depth -= 1
            current_string.append(c)
            if current_depth == 0:
                result.append(''.join(current_string))
                current_string = []
    return result

Example 3:
def truncate_number(number: float, decimals: int) -> float:
    \"\"\" Truncate a floating point number to a specified number of decimal places.
    >>> truncate_number(3.14159, 2)
    3.14
    >>> truncate_number(2.71828, 1)
    2.7
    \"\"\"
    import math
    factor = 10 ** decimals
    return math.trunc(number * factor) / factor

Example 4:
def below_zero(operations: List[Tuple[str, int]], case_insensitive: bool = False) -> bool:
    \"\"\" You're given a list of deposit and withdrawal operations on a bank account that starts with
    zero balance. Your task is to detect if at any point the balance of account falls below zero.
    >>> below_zero([('Deposit', 1), ('Deposit', 2), ('Withdrawal', 4)])
    True
    >>> below_zero([('Deposit', 1), ('Deposit', 2), ('withdrawal', 3)], True)
    True
    \"\"\"
    balance = 0
    for op in operations:
        if case_insensitive:
            op_name = op[0].lower()
        else:
            op_name = op[0]
        if op_name == "deposit":
            balance += op[1]
        else:
            balance -= op[1]
        if balance < 0:
            return True
    return False

"""


def strip_thinking_blocks(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>", "", text)
    text = re.sub(r"</think>", "", text)
    text = re.sub(r"◇[^◇]*?◇", "", text, flags=re.DOTALL)
    text = re.sub(r"◇", "", text)
    return text.strip()


def check_correctness_humaneval(prompt: str, generated: str, correct: str) -> bool:
    generated = strip_thinking_blocks(generated)
    body = generated.strip()
    if not body:
        return False

    body = re.sub(r"^```python\n?", "", body)
    body = re.sub(r"^```\n?", "", body)
    body = re.sub(r"```$", "", body)
    body = body.strip()

    if not body.startswith("def ") and not body.startswith("    ") and not body.startswith("\t"):
        lines = body.split("\n")
        code_start = None
        for i, line in enumerate(lines):
            stripped = line.strip()
            if (stripped.startswith("def ") or
                stripped.startswith("return ") or
                stripped.startswith("if ") or
                stripped.startswith("for ") or
                stripped.startswith("while ") or
                stripped.startswith("import ") or
                stripped.startswith("from ") or
                stripped.startswith("class ") or
                stripped.startswith("try ") or
                stripped.startswith("with ") or
                stripped.startswith("@") or
                stripped.startswith("#") or
                line.startswith("    ") or
                line.startswith("\t")):
                code_start = i
                break
        if code_start is not None:
            body = "\n".join(lines[code_start:])
        else:
            return body.strip() == correct.strip()

    try:
        compile(body, "<string>", "exec")
        full_code = body
    except SyntaxError:
        sig_match = re.search(
            r"^(def\s+\w+\s*\([^)]*\)\s*(->\s*\w+\s*)?:\s*)",
            prompt,
            re.MULTILINE,
        )
        if not sig_match:
            return body.strip() == correct.strip()

        signature = sig_match.group(1).strip()
        docstring_match = re.search(r'(""".*?""")', prompt, re.DOTALL)
        docstring = docstring_match.group(1) if docstring_match else '""""""'

        full_code = (
            f"{signature}\n    {docstring}\n{textwrap.indent(body, '    ')}\n"
        )

    try:
        compile(full_code, "<string>", "exec")
    except SyntaxError:
        return False

    namespace = {}
    try:
        exec(full_code, namespace)
    except Exception:
        return False

    docstring_match = re.search(r'(""".*?""")', prompt, re.DOTALL)
    docstring = docstring_match.group(1) if docstring_match else '""""""'
    examples = re.findall(
        r">>>\s*(.+?)\n\s*(.+?)(?=\n\s*>>>|\n\s*\"\"\"|$)",
        docstring,
        re.DOTALL,
    )
    if not examples:
        return body.strip() == correct.strip()

    passed = 0
    for call_str, expected_str in examples:
        call_str = call_str.strip()
        expected_str = expected_str.strip()
        try:
            result = eval(call_str, namespace)
            if repr(result) == expected_str:
                passed += 1
        except Exception:
            pass

    return passed >= max(1, len(examples) // 2)


def apply_chat_template(user_content: str, tokenizer, enable_thinking: bool = True) -> str:
    messages = [{"role": "user", "content": user_content}]
    chat_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    # Thinking suffix: activates Qwen's internal reasoning for complex tasks
    # Non-thinking: plain assistant prompt for simple classification/code generation
    if enable_thinking:
        return chat_prompt + "<|im_start|>assistant\n\n"
    else:
        return chat_prompt + "<|im_start|>assistant\n\n"


def format_humaneval_prompt(raw_prompt: str) -> str:
    return (
        f"{HUMANEVAL_FEWSHOT}"
        "Now complete this function:\n\n"
        f"{raw_prompt}"
    )


def load_humaneval_questions(n: int) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("openai_humaneval", split=f"test[:{min(n, 164)}]")
    questions = []
    for i, item in enumerate(ds):
        questions.append({
            "question_id": f"{item['task_id'].replace('/', '_')}",
            "dataset": "humaneval",
            "prompt": item["prompt"],
            "correct_answer": item["canonical_solution"],
        })
    return questions


def assign_splits(question_ids: list[str], seed: int = 42) -> dict[str, str]:
    import numpy as np

    rng = np.random.RandomState(seed)
    splits = rng.choice(["train", "val"], size=len(question_ids), p=[0.8, 0.2])
    return {qid: split for qid, split in zip(question_ids, splits)}


@app.function(
    gpu="L4",
    volumes={"/vol": volume},
    image=image,
    timeout=10800,
)
def extract_multilayer_humaneval(
    layers: list[int],
    n_samples: int,
    max_new_tokens: int,
) -> dict:
    import numpy as np
    import torch
    import torch.nn.functional as F
    from tqdm import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    def _log_gpu_memory(label: str):
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"  [GPU {label}] Allocated: {allocated:.2f} GiB | Reserved: {reserved:.2f} GiB | Total: {total:.2f} GiB")

    print("=" * 60)
    print("MULTILAYER HUMANEVAL EXTRACTION")
    print(f"Layers: {layers}")
    print(f"Samples: {n_samples}")
    print(f"Max new tokens: {max_new_tokens}")
    print("=" * 60)
    _log_gpu_memory("start")

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
    tokenizer.padding_side = "left"
    print(f"  Model loaded on {next(model.parameters()).device}")
    _log_gpu_memory("after model load")

    print("\n── Loading HumanEval dataset ──")
    questions = load_humaneval_questions(n_samples)
    print(f"  Loaded {len(questions)} questions")

    question_ids = [q["question_id"] for q in questions]
    split_map = assign_splits(question_ids, seed=SEED)

    out_dir = Path(RESULTS_DIR) / "multilayer_humaneval"
    act_dir = out_dir / "activations"
    act_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "humaneval_metadata.jsonl"
    split_path = out_dir / "humaneval_split.json"

    with open(meta_path, "w") as _:
        pass

    with open(split_path, "w") as f:
        json.dump(split_map, f, indent=2)
    print(f"  Split assignments saved to {split_path}")

    results = []
    num_batches = (len(questions) + BATCH_SIZE - 1) // BATCH_SIZE
    batch_iter = tqdm(
        range(0, len(questions), BATCH_SIZE),
        total=num_batches,
        desc="HumanEval batches",
    )

    start_time = time.time()

    for batch_start in batch_iter:
        batch = questions[batch_start:batch_start + BATCH_SIZE]
        user_contents = [format_humaneval_prompt(q["prompt"]) for q in batch]
        chat_prompts = [apply_chat_template(uc, tokenizer, enable_thinking=False) for uc in user_contents]

        inputs = tokenizer(
            chat_prompts, return_tensors="pt", padding=True, truncation=True
        ).to(model.device)
        input_len = inputs.input_ids.shape[1]

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True,
            output_hidden_states=True,
            suppress_tokens=[248068],
        )
        with torch.no_grad():
            gen_outputs = model.generate(**inputs, **gen_kwargs)
        _log_gpu_memory("after generate")

        # Extract prefill hidden states from the first generation step
        # gen_outputs.hidden_states[0] is a tuple of (batch, seq_len, hidden_dim) per layer
        prefill_hidden_states = gen_outputs.hidden_states[0]
        batch_hidden = {}
        for layer_idx in layers:
            hs = prefill_hidden_states[layer_idx][:, -1, :].cpu().numpy()
            batch_hidden[layer_idx] = hs

        sequences = gen_outputs.sequences
        scores = gen_outputs.scores
        gen_ids = sequences[:, input_len:]
        gen_len = gen_ids.shape[1]

        for idx, q in enumerate(batch):
            qid = q["question_id"]

            for layer_idx in layers:
                hs = batch_hidden[layer_idx][idx]
                hs = hs.reshape(1, HIDDEN_DIM)
                np.save(act_dir / f"{qid}__layer_{layer_idx}.npy", hs)

            sample_gen_ids = gen_ids[idx]
            eos_mask = sample_gen_ids == tokenizer.eos_token_id
            eos_positions = eos_mask.nonzero(as_tuple=True)[0]
            if len(eos_positions) > 0:
                actual_len = eos_positions[0].item() + 1
            else:
                actual_len = gen_len

            token_msps = []
            token_entropies = []
            for t in range(actual_len):
                logits_t = scores[t][idx]
                probs_t = F.softmax(logits_t, dim=-1)
                msp_t = probs_t.max().item()
                entropy_t = -(probs_t * torch.log(probs_t + 1e-12)).sum().item()
                token_msps.append(msp_t)
                token_entropies.append(entropy_t)

            mean_msp = float(np.mean(token_msps)) if token_msps else 0.0
            mean_entropy = float(np.mean(token_entropies)) if token_entropies else 0.0

            generated_ids = sample_gen_ids[:actual_len]
            generated_text = tokenizer.decode(
                generated_ids, skip_special_tokens=True
            )

            correct = check_correctness_humaneval(
                q["prompt"], generated_text, q["correct_answer"]
            )

            result = {
                "id": qid,
                "msp": mean_msp,
                "entropy": mean_entropy,
                "generated_text": generated_text,
                "correctness": correct,
                "split": split_map[qid],
            }
            results.append(result)
            with open(meta_path, "a") as f:
                f.write(json.dumps(result, default=float) + "\n")

            n_processed = len(results)
            n_correct = sum(1 for r in results if r["correctness"])
            running_acc = n_correct / n_processed if n_processed > 0 else 0.0
            print(f"  [{qid}] {'✓' if correct else '✗'} | MSP={mean_msp:.4f} | Ent={mean_entropy:.4f} | Running: {n_correct}/{n_processed} = {running_acc:.1%}")

    elapsed = time.time() - start_time
    n_correct = sum(1 for r in results if r["correctness"])
    accuracy = n_correct / len(results) if results else 0.0

    print("\n" + "=" * 60)
    print("EXTRACTION COMPLETE")
    print("=" * 60)
    print(f"  Questions: {len(results)}")
    print(f"  Correct:   {n_correct} ({accuracy:.1%})")
    print(f"  Train:     {sum(1 for r in results if r['split'] == 'train')}")
    print(f"  Val:       {sum(1 for r in results if r['split'] == 'val')}")
    print(f"  Time:      {elapsed:.1f}s")
    print(f"  Output:    {out_dir}")
    print("=" * 60)

    return {
        "n_questions": len(results),
        "n_correct": n_correct,
        "accuracy": accuracy,
        "elapsed_seconds": elapsed,
        "output_dir": str(out_dir),
    }


@app.local_entrypoint()
def main(
    layers: str = "15,17,19,20,22,25",
    n_samples: int = DEFAULT_N_SAMPLES,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
):
    layer_list = [int(x.strip()) for x in layers.split(",")]
    print(f"Launching extraction: layers={layer_list}, n={n_samples}, max_new={max_new_tokens}")
    result = extract_multilayer_humaneval.remote(layer_list, n_samples, max_new_tokens)
    print(f"\nRemote result: {result}")


def _validate_prompts_local(n_samples: int, layers: list[int]) -> None:
    import sys

    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen3.5-4B",
            trust_remote_code=True,
        )
    except Exception as exc:
        print(f"ERROR: Could not load tokenizer: {exc}", file=sys.stderr)
        print("Make sure 'transformers' is installed and you have internet access.", file=sys.stderr)
        sys.exit(1)

    print("=" * 60)
    print("PROMPT VALIDATION (first 5 samples)")
    print(f"Layers target: {layers}")
    print(f"N samples requested: {n_samples}")
    print("=" * 60)

    questions = load_humaneval_questions(n_samples)
    for i, q in enumerate(questions[:5]):
        user_content = format_humaneval_prompt(q["prompt"])
        chat_prompt = apply_chat_template(user_content, tokenizer)

        print(f"\n{'─' * 60}")
        print(f"Sample {i + 1} — {q['question_id']}")
        print(f"{'─' * 60}")
        print(chat_prompt)
        print(f"{'─' * 60}")
        print(f"Prompt length: {len(chat_prompt)} chars")
        print(f"User content length: {len(user_content)} chars")

    print("\n" + "=" * 60)
    print("VALIDATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Multi-layer HumanEval extraction with MSP/entropy"
    )
    parser.add_argument(
        "--layers",
        default="15,17,19,20,22,25",
        help="Comma-separated layer indices to extract (default: 15,17,19,20,22,25)",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=DEFAULT_N_SAMPLES,
        help="Number of HumanEval samples (default: 100)",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help="Max new tokens per generation (default: 2048)",
    )
    parser.add_argument(
        "--validate-prompts",
        action="store_true",
        help="Run preprocessing pipeline on first 5 prompts without GPU inference",
    )
    args = parser.parse_args()

    layer_list = [int(x.strip()) for x in args.layers.split(",")]

    if args.validate_prompts:
        _validate_prompts_local(args.n_samples, layer_list)
    else:
        print(
            "This script is designed to run on Modal GPU.\n"
            "To execute (detached):\n"
            "  modal run --detach scripts/extract_multilayer_humaneval.py\n"
            "\n"
            "To validate prompts locally (no GPU):\n"
            "  python scripts/extract_multilayer_humaneval.py --validate-prompts\n"
        )
