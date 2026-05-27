# Learnings — Benchmark Re-Extraction

## 2026-05-19 Session Start

### Key Context
- Qwen3.5-4B base model activations are BROKEN (zero-shot, no chat_template, 256 tokens)
- Root cause: model treated as base, not instruct → near-random outputs
- Correct pattern: `tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)` + thinking suffix
- T8 GSM8K script has the gold standard pattern (lines 59-69)
- 4 critical bugs: no chat_template, low token limits, zero-shot, only 50 samples

### Research-Backed Accuracy Targets
- MATH: 45-55% (Qwen3-4B = 54.1%)
- HumanEval: 40-60% (Qwen3-4B = 67%)
- ARC-Challenge: 60-80% (Phi-3.5-mini = 84.6%)
- TriviaQA: 55-70% (Gemma 3n E4B = 70.2%)

### GPU Decision
- L4 at $0.80/hr is optimal (1.5-2× faster than T4, lower total cost)
- Budget: $9.44 remaining, cap at $8.00

### File References
- T8 correct pattern: `scripts/gen_time_extract_qwen_prompting.py:59-69`
- Correctness checkers: `scripts/combined_modal_extraction.py:128-180`
- Evaluation pipeline: `scripts/cross_benchmark_eval.py:317-353`
- Modal config: volume `epistemic-model-cache`, model at `/vol/models/Qwen_Qwen3.5-4B`

### Conventions
- Activation files named `{question_id}__layer_25.npy`
- Saved in `data/benchmark_activations/{benchmark}/`
- Results in `data/benchmark_activations/{benchmark}_results.jsonl`
## 2026-05-20 Thinking Block Fix

### Problem
- Qwen3.5-4B outputs `<think>...</think>` blocks even when `enable_thinking=False`
- HumanEval: 0% accuracy (checker tried to compile thinking text as Python)
- TriviaQA: 33% accuracy (substring match failed when thinking text overwhelmed answer)
- ARC: 90% but fragile — model can still emit thinking blocks with thinking=OFF

### Fix Applied
- Added `strip_thinking_blocks(text) -> str` helper (handles `<think>...</think>` and `◇...◇` blocks)
- Integrated into `check_correctness_humaneval()`, `check_correctness_triviaqa()`, `check_correctness_arc()`, and `extract_answer_arc()`
- Left `extract_answer_math()` unchanged (it already strips thinking blocks inline)
- Left `_apply_chat_template()` and thinking mode assignments untouched

### Key Insight
- `enable_thinking=False` only changes the prompt suffix (`assistant\n` vs `assistant\n\n`)
- It does NOT suppress the model's internal reasoning — the model still generates `<think>` blocks
- All correctness checkers must strip thinking blocks BEFORE processing, regardless of the thinking mode flag

### Verification
- Syntax validated with `python3 -c "import ast; ast.parse(...)"` → OK
- File: `scripts/re_extract_benchmarks.py`
