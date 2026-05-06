"""Generation-time hidden state extraction on Modal.

Runs Qwen3.5-4B on Modal T4 GPU, generates CoT responses for all 656 questions,
and extracts hidden states at every 5th token during generation at layer 30.

Cost estimate:
    656 questions x ~15-30 s generation each ≈ 2.5-5.5 hours on T4
    T4 cost: $0.000164/sec ≈ $0.59/hr
    Estimated total: $1.50-$3.25
    Budget ceiling: $8

Output:
    Saves to Modal volume ``epistemic-model-cache`` under ``results/gen_time/``:
    - gen_time_batch_{NNNN}.pkl : intermediate checkpoints every 50 questions
    - gen_time_all.pkl          : final consolidated result
"""

import modal
from modal import App, Volume, Image

app = App("gen-time-extraction")

# Use the existing volume with model and prior results
volume = Volume.from_name("epistemic-model-cache")
MODEL_DIR = "/vol/model"

image = (
    Image.debian_slim()
    .pip_install("torch", "transformers", "numpy", "tqdm")
)


@app.function(
    gpu="T4",
    volumes={"/vol": volume},
    image=image,
    timeout=21_600,  # 6 hours — conservative for 656 questions on T4
)
def extract_all():
    import time
    import json
    import pickle
    from pathlib import Path

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import numpy as np
    from tqdm import tqdm

    start_time = time.time()

    # ── Load model ──────────────────────────────────────────────────────────
    print("Loading model from volume ...")
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

    print(f"Model loaded on {next(model.parameters()).device}")

    # ── Load questions ──────────────────────────────────────────────────────
    print("Loading questions from volume ...")
    questions_path = Path("/vol/results/probe_extract_results.jsonl")
    with open(questions_path, "r") as f:
        questions = [json.loads(line) for line in f if line.strip()]
    print(f"Loaded {len(questions)} questions")

    # ── Setup output ────────────────────────────────────────────────────────
    out_dir = Path("/vol/results/gen_time")
    out_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    target_layer = 30
    sample_every = 5

    # ── Main loop ───────────────────────────────────────────────────────────
    for i, q in enumerate(tqdm(questions, desc="Generating")):
        prompt = q["prompt"]

        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids.to(model.device)
        attention_mask = inputs.attention_mask.to(model.device)

        gen_hidden_states: list[np.ndarray] = []
        step_counter = [0]  # mutable closure — first call is prefill

        def hook_fn(module, input, output):
            """Capture hidden state of the last token at each forward pass."""
            step_counter[0] += 1
            if step_counter[0] == 1:
                return  # skip prefill; we only want generation-time states
            # output[0] shape: [batch, seq_len, hidden_dim]
            # During generation with KV-cache seq_len is 1.
            hs = output[0][:, -1, :].detach().cpu().numpy()
            gen_hidden_states.append(hs)

        layer = model.model.layers[target_layer]
        handle = layer.register_forward_hook(hook_fn)

        try:
            with torch.no_grad():
                output_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=256,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )

            # Decode only the newly generated tokens
            generated_text = tokenizer.decode(
                output_ids[0][input_ids.shape[1] :],
                skip_special_tokens=True,
            )

            # Sample every 5th generation token
            sampled_hs = gen_hidden_states[::sample_every]
            token_positions = list(range(0, len(gen_hidden_states), sample_every))

            result = {
                "question_id": q["question_id"],
                "dataset": q["dataset"],
                "subject": q.get("subject"),
                "hidden_states": sampled_hs,  # list of (1, hidden_dim) arrays
                "token_positions": token_positions,
                "generated_text": generated_text,
                "correct": q["correct"],
                "n_generated_tokens": len(gen_hidden_states),
            }
            results.append(result)

        except Exception as exc:
            print(f"\nERROR on {q['question_id']}: {exc}")
            results.append(
                {
                    "question_id": q["question_id"],
                    "dataset": q["dataset"],
                    "error": str(exc),
                    "correct": q["correct"],
                }
            )
        finally:
            handle.remove()

        # Intermediate checkpoint every 50 questions
        if (i + 1) % 50 == 0:
            batch_path = out_dir / f"gen_time_batch_{i + 1:04d}.pkl"
            with open(batch_path, "wb") as f:
                pickle.dump(results, f)
            elapsed = time.time() - start_time
            print(
                f"\nCheckpoint {i + 1}/{len(questions)} — "
                f"elapsed {elapsed / 60:.1f} min"
            )

    # ── Final save ──────────────────────────────────────────────────────────
    final_path = out_dir / "gen_time_all.pkl"
    with open(final_path, "wb") as f:
        pickle.dump(results, f)

    elapsed = time.time() - start_time
    cost_usd = elapsed * 0.000164

    print("\n" + "=" * 60)
    print(f"Done. {len(results)}/{len(questions)} questions processed.")
    print(f"Total GPU time: {elapsed / 60:.1f} min  ({elapsed / 3600:.2f} hr)")
    print(f"Estimated Modal cost: ${cost_usd:.2f}")
    print("=" * 60)

    return {
        "n_questions": len(results),
        "elapsed_seconds": elapsed,
        "estimated_cost_usd": cost_usd,
    }


@app.local_entrypoint()
def main():
    result = extract_all.remote()
    print(f"Remote result: {result}")
