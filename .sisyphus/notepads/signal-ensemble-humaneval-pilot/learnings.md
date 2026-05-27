# Signal Ensemble - Humaneval Pilot Learnings

## Task 1: Create 5 pytest test files for signal ensemble logic

### Context
- Repo: /Users/aban/drive/Projects/epistemic-steering
- Hidden size: 2560, Layers: 32, Model: Qwen3.5-4B
- Existing test framework: pytest with class-based organization (no conftest.py)
- Two import patterns: `sys.path.insert` + direct, or `from src.module import ...` (works due to `pythonpath = ["."]` in pyproject.toml)
- 199 total tests pass, 1 pre-existing skip (test_prompt_deduplication)

### Files Created

**Source modules (src/):**
1. `src/lvu_parser.py` — LVU score via hedge words (maybe, I think, could be) and self-correction patterns (wait, actually, hold on)
2. `src/hybrid_backoff.py` — z-score normalization, alpha=0.7 in-domain / alpha=0.0 on OOD, hybrid formula
3. `src/ood_detector.py` — MSP gap thresholds: HIGH (>0.3), MODERATE (0.1-0.3), LOW (≤0.1)
4. `src/mean_pool.py` — mean pooling (seq_len, hidden_dim) → (hidden_dim,), empty → zero vector
5. `src/extraction_schema.py` — schema validation for extraction dict with pre-registered layers [15, 17, 19, 20, 25]

**Test files (tests/):**
1. `tests/test_lvu_parser.py` — 17 assertions (hedge detection ×6, self-correction ×5, lvu_score ×6)
2. `tests/test_hybrid_backoff.py` — 12 assertions (zscore ×5, alpha ×3, hybrid_score ×4)
3. `tests/test_ood_detector.py` — 7 assertions (alarm levels ×6)
4. `tests/test_mean_pool.py` — 7 assertions (shape, values, empty, error ×6)
5. `tests/test_extraction_schema.py` — 14 assertions (constants ×3, validation ×11)

**Total new assertions: 57** (requirement was ≥20)

### Key Design Decisions
- lvu_score: hedge words contribute 0.5, self-correction contributes 0.5, capped at 1.0
- Empty/missing text → lvu_score = 1.0 (maximum uncertainty)
- zscore_normalize: std=0 or n≤1 returns all zeros (safe edge handling)
- compute_alpha: only depends on ood_alarm flag, probe_score kept for API compatibility
- mean_pool: ValueError on non-2D input, zero vector on empty sequence
- extraction_schema: layer arrays must be np.ndarray with shape (1, 2560), split in ("train", "val")
- Followed existing class-based test organization (TestXxx with descriptive method names)
- No GPU, sklearn, or transformers dependencies in test imports

### Gotchas
- Source modules didn't exist — created minimal implementations alongside tests
- pyproject.toml `pythonpath = ["."]` enables `from src.module import ...` pattern
- Mean pool empty sequence: shape[0]=0 → np.zeros(shape[1]) (must preserve dtype)
- Hybrid score z-score normalizes all 3 signals together (same mean/std context)

## Task 4: Create extract_multilayer_humaneval.py

### Script Created
- `scripts/extract_multilayer_humaneval.py` — 510 lines

### Key Features Implemented
- `--layers 15,17,19,20,22,25` argument parsing (comma-separated ints)
- Multi-layer hidden state extraction during single forward pass via `output_hidden_states=True`
- MSP per sample: mean of `max(softmax(logits_t))` across all generated tokens (trimmed to first EOS)
- Entropy per sample: mean of `-sum(p * log(p))` across all generated tokens
- Generated text saved in output JSONL for downstream LVU parsing
- Random 80/20 train/val split (seed=42), NOT stratified, saved to `humaneval_split.json`
- Qwen3.5 chat_template with thinking mode (`<think>\n` suffix), no system prompt injection
- 4-shot CoT examples inlined for HumanEval (has_close_elements, separate_paren_groups, truncate_number, below_zero)
- `--validate-prompts` flag: loads tokenizer from HF hub locally, prints first 5 preprocessed prompts to stdout
- Correctness check via inlined `check_correctness_humaneval` from `re_extract_benchmarks.py`
- Budget: T4 GPU ($0.59/hr)

### Design Decisions
- Dual-mode script: argparse for local `--validate-prompts`, Modal `local_entrypoint` for GPU extraction
- Output structure per sample: `{"id": str, "msp": float, "entropy": float, "generated_text": str, "correctness": bool, "split": str}`
- Hidden states saved as individual `.npy` files per layer per sample: `{qid}__layer_{idx}.npy` shaped `(1, 2560)`
- Metadata saved as JSONL: `humaneval_metadata.jsonl`
- Split assignments saved as JSON: `humaneval_split.json`
- Actual generation length determined by first EOS token to avoid biasing MSP/entropy with repeated EOS tokens

### Verification
- `python scripts/extract_multilayer_humaneval.py --validate-prompts --n_samples 1` passes
- Prompt output shows correct chat_template formatting with `<|im_start|>user/assistant` and `<think>` trigger
- Local tokenizer loaded from `Qwen/Qwen3.5-4B` (cached HF hub model)

### Gotchas
- The HF hub model name for local tokenizer loading is `Qwen/Qwen3.5-4B`, not `Qwen/Qwen3.5-4B-Instruct`
- `model.generate()` with `return_dict_in_generate=True, output_scores=True` returns `scores` as a tuple of `(batch, vocab_size)` tensors, one per generated token
- Hidden states tuple indexing: index 0 = embeddings, index 1-32 = transformer layers. So layer 25 (human) = index 25.
