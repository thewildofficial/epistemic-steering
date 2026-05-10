"""Quick one-question test to verify model.generate() + forward hook pattern.
Checks: generation quality, hidden state capture count, token alignment.
"""
from __future__ import annotations
import modal
from modal import App, Image, Volume

app = App("gen-time-test")
volume = Volume.from_name("epistemic-model-cache")
MODEL_DIR = "/vol/models/Qwen_Qwen3.5-4B"
RESULTS_DIR = "/vol/results"
FEWSHOT_PATH = f"{RESULTS_DIR}/gsm8k_fewshot_prompt.txt"
LAYER = 25

image = (
    Image.debian_slim()
    .pip_install("torch", "transformers", "numpy", "tqdm", "accelerate")
)

@app.function(image=image, volumes={"/vol": volume}, gpu="A100", timeout=600)
def test_generation():
    import torch, numpy as np
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    ).eval()
    print(f"Model loaded on {next(model.parameters()).device}")

    with open(FEWSHOT_PATH) as f:
        fewshot = f.read()

    question = "James buys a train ticket for $175. The ticket price includes a 25% summer surcharge. What was the original price of the ticket before the surcharge?"
    
    messages = [{"role": "user", "content": f"{fewshot}\nQuestion: {question}\nLet's think step by step"}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    text = text + "<think>\n"

    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs.input_ids.to(model.device)
    attention_mask = inputs.attention_mask.to(model.device)
    prefill_len = input_ids.shape[1]
    print(f"Prefill length: {prefill_len} tokens")

    hidden_states_per_token = []
    step_counter = [0]

    def hook_fn(module, input, output):
        step_counter[0] += 1
        if step_counter[0] == 1:  # skip prefill
            return
        h = output[0] if isinstance(output, tuple) else output
        hs = h[:, -1, :].cpu().float().numpy()
        hidden_states_per_token.append(hs[0])

    layer = model.model.layers[LAYER]
    handle = layer.register_forward_hook(hook_fn)

    try:
        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=2048,
                do_sample=True,
                temperature=0.6,
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
    finally:
        handle.remove()

    generated_ids = output_ids[0, prefill_len:]
    n_generated = len(generated_ids)
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    n_hs = len(hidden_states_per_token)

    print(f"\n{'='*60}")
    print(f"VERIFICATION RESULTS")
    print(f"{'='*60}")
    print(f"Hook calls (total):        {step_counter[0]}")
    print(f"Generated tokens:          {n_generated}")
    print(f"Hidden states captured:    {n_hs}")
    print(f"Match (hs == gen_tokens):  {n_hs == n_generated}")
    print(f"hs shape:                  {hidden_states_per_token[0].shape if n_hs else 'N/A'}")
    print(f"\n{'='*60}")
    print(f"GENERATED TEXT (first 500 chars)")
    print(f"{'='*60}")
    print(generated_text[:500])
    if len(generated_text) > 500:
        print(f"... (truncated, total {len(generated_text)} chars)")
    print(f"\n{'='*60}")
    print(f"ANSWER EXTRACTION")
    print(f"{'='*60}")
    import re
    # Try to find the answer in the generated text
    if "The answer is" in generated_text:
        answer_section = generated_text.split("The answer is")[-1][:200]
        print(f"Answer section: {answer_section.strip()}")
    numbers = re.findall(r'-?\d+\.?\d*', generated_text)
    print(f"All numbers found: {numbers[-5:] if len(numbers) > 5 else numbers}")

    return {
        "n_prefill": prefill_len,
        "n_generated": n_generated,
        "n_hidden_states": n_hs,
        "match": n_hs == n_generated,
        "text_preview": generated_text[:300],
    }

@app.local_entrypoint()
def main():
    result = test_generation.remote()
    print(f"\nRemote result: {result}")
