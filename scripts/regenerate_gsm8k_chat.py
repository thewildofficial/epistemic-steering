"""Regenerate GSM8K answers with proper chat template prompting on Modal.

Uses Qwen3.5-4B-Instruct with tokenizer.apply_chat_template() to fix the
3.5% accuracy issue caused by raw text prompting.

Also extracts prefill hidden states at every layer for probe training.

Speed optimization:
    BATCH_SIZE=4 for generation (3-4x faster than sequential)
    max_new_tokens=1024 (GSM8K CoT typically 200-600 tokens)
    Full generated_text saved for offline extraction (more reliable)

Cost estimate:
    ~200 GSM8K questions × ~7.5s each (batched) ≈ 25 minutes on T4
    T4 cost: $0.000164/sec ≈ $0.59/hr
    Estimated total: $0.25 (plus overhead)
    Budget ceiling: $5

Output:
    Saved to Modal volume ``epistemic-model-cache`` under ``results/gsm8k_chat/``:
    - gsm8k_chat_results.jsonl  : generation results with accuracy
    - activations/              : per-question, per-layer prefill hidden states

Note: answers are extracted offline from saved generated_text for reliability.
"""

import modal
from modal import App, Volume, Image

app = App("gsm8k-regenerate")

volume = Volume.from_name("epistemic-model-cache")
MODEL_PATH = "/vol/models/Qwen_Qwen3.5-4B"

image = (
    Image.debian_slim()
    .pip_install("torch", "transformers", "numpy", "tqdm", "accelerate")
)


@app.function(
    gpu="T4",
    volumes={"/vol": volume},
    image=image,
    timeout=7_200,
)
def regenerate_gsm8k():
    import time
    import json
    import re
    import numpy as np
    from pathlib import Path

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from tqdm import tqdm

    start_time = time.time()

    print("Loading model from volume ...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Model loaded on {next(model.parameters()).device}")

    print("Loading questions from volume ...")
    with open("/vol/results/probe_extract_results.jsonl", "r") as f:
        all_qs = [json.loads(line) for line in f if line.strip()]

    gsm8k = [q for q in all_qs if q["dataset"] == "gsm8k"]
    print(f"Found {len(gsm8k)} GSM8K questions (of {len(all_qs)} total)")

    out_dir = Path("/vol/results/gsm8k_chat")
    out_dir.mkdir(parents=True, exist_ok=True)
    act_dir = out_dir / "activations"
    act_dir.mkdir(parents=True, exist_ok=True)

    BATCH_SIZE = 4

    results = []
    n_layers = model.config.num_hidden_layers

    for batch_start in range(0, len(gsm8k), BATCH_SIZE):
        batch_qs = gsm8k[batch_start:batch_start + BATCH_SIZE]
        batch_size = len(batch_qs)

        # Build chat inputs for this batch
        chat_inputs = []
        for q in batch_qs:
            messages = [{"role": "user", "content": q["prompt"]}]
            chat_input = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            chat_inputs.append(chat_input)

        # Tokenize with padding
        padded = tokenizer(
            chat_inputs, return_tensors="pt", padding=True
        ).to(model.device)
        input_ids = padded.input_ids
        attention_mask = padded.attention_mask

        # Prefill: one forward pass per question to get hidden states
        prefill_outputs_list = []
        for j in range(batch_size):
            with torch.no_grad():
                out = model(
                    input_ids=input_ids[j:j+1],
                    attention_mask=attention_mask[j:j+1],
                    output_hidden_states=True,
                    use_cache=True,
                )
            prefill_outputs_list.append(out)

        # Extract prefill states for each question
        for j, (q, out) in enumerate(zip(batch_qs, prefill_outputs_list)):
            prefill_states = {}
            for layer_idx in range(n_layers):
                hs = out.hidden_states[layer_idx + 1][:, -1, :].cpu().numpy()
                prefill_states[str(layer_idx)] = hs
            act_path = act_dir / f"{q['question_id']}_prefill.npz"
            np.savez(act_path, **prefill_states)

        # Batch generation - reuse past_key_values from prefill
        with torch.no_grad():
            gen_outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=[o.past_key_values for o in prefill_outputs_list],
                max_new_tokens=1024,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Decode and extract for each question
        for j, (q, gen_out) in enumerate(zip(batch_qs, gen_outputs)):
            full_text = tokenizer.decode(gen_out, skip_special_tokens=True)
            input_text = tokenizer.decode(input_ids[j], skip_special_tokens=True)
            generated = full_text[len(input_text):].strip()

            # Extract answer: search from end for a line containing a number
            lines = generated.strip().split('\n')
            extracted = None
            for line in reversed(lines):
                line = line.strip()
                nums = re.findall(r'\b(\d+(?:\.\d+)?)\b', line)
                if nums:
                    extracted = nums[-1]
                    break
            extracted_answer = extracted if extracted is not None else "?"

            correct = str(extracted_answer).strip() == str(q["correct_answer"]).strip()

            result = {
                "question_id": q["question_id"],
                "dataset": "gsm8k",
                "prompt": q["prompt"],
                "chat_formatted": chat_inputs[j][:200] + "..." if len(chat_inputs[j]) > 200 else chat_inputs[j],
                "generated_text": generated,
                "correct_answer": q["correct_answer"],
                "model_answer": extracted_answer,
                "correct": correct,
            }
            results.append(result)

        # Progress checkpoint every batch
        batch_idx = batch_start + batch_size
        acc = sum(1 for r in results if r["correct"]) / len(results)
        print(
            f"Progress: {batch_idx}/{len(gsm8k)}, "
            f"accuracy: {acc:.1%} ({sum(1 for r in results if r['correct'])}/{len(results)})"
        )
        checkpoint_path = out_dir / f"gsm8k_chat_checkpoint_{batch_idx:04d}.jsonl"
        with open(checkpoint_path, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")

    correct_count = sum(1 for r in results if r["correct"])
    total = len(results)
    accuracy = 100 * correct_count / total if total > 0 else 0

    print("\n" + "=" * 60)
    print(f"Final accuracy: {correct_count}/{total} = {accuracy:.1f}%")
    print("=" * 60)

    results_path = out_dir / "gsm8k_chat_results.jsonl"
    with open(results_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    elapsed = time.time() - start_time
    cost_usd = elapsed * 0.000164

    print(f"Saved results to {results_path}")
    print(f"Saved activations to {act_dir}")
    print(f"Total GPU time: {elapsed / 60:.1f} min  ({elapsed / 3600:.2f} hr)")
    print(f"Estimated Modal cost: ${cost_usd:.2f}")

    return {
        "n_questions": total,
        "correct": correct_count,
        "accuracy": accuracy,
        "elapsed_seconds": elapsed,
        "estimated_cost_usd": cost_usd,
    }


@app.local_entrypoint()
def main():
    result = regenerate_gsm8k.remote()
    print(f"\nRemote result: {result}")

    import subprocess

    local_dir = Path("data/gsm8k_chat")
    local_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nDownloading results to {local_dir.resolve()} ...")
    cmd = [
        "modal", "volume", "get", "epistemic-model-cache",
        "results/gsm8k_chat/", str(local_dir), "--recursive",
    ]
    subprocess.run(cmd, check=False)
    print("Download complete.")
