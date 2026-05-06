# Epistemic Steering Learnings

## Task 2: Data Loading + Validation Pipeline

### Context
- Repo at /Users/aban/drive/Projects/epistemic-steering
- UV package manager: `uv run python`, `uv run pytest`
- Model: Qwen3.5-4B, 32 layers, 2560 hidden dim
- 656 questions total: 456 MMLU + 200 GSM8K
- 199 MMLU questions have model_answer="?" — treat as incorrect

### Decisions Made
- Using pandas for DataFrame operations (added to pyproject.toml)
- Using numpy for activation array loading
- JSONL format: one JSON object per line
- Path template for activations: `{path}/q{question_id}_layer{layer}.npy`

### Patterns
- validate_data returns dict with missing_counts, dataset_sizes, errors
- clean_mmlu_answers modifies DataFrame in-place, doesn't drop rows
- split_by_dataset returns tuple of (mmlu_df, gsm8k_df)

### Gotchas
- correct column is bool type, not string "True"/"False"
- token_positions column is list of ints in allpos results
- Question IDs are ints, not zero-padded strings in paths
- Empty JSONL file causes KeyError when accessing columns - handle with `if df.empty: return df`
- validate_data needs explicit error for missing values (not just count)
- PYTHONPATH must be set when running pytest (`PYTHONPATH=. uv run pytest`)
## Task 4: Evaluation Metrics (evaluate.py)

### Context
- Implemented 7 functions in src/evaluate.py for hallucination detection metrics
- Created comprehensive test suite in tests/test_evaluate.py with 40 tests

### Decisions Made
- Labels array converted to bool explicitly with `np.asarray(labels, dtype=bool)` to handle empty arrays and avoid TypeError in bitwise operations
- Used sklearn.calibration_curve with strategy='uniform' as specified
- compute_all_metrics aggregates all other functions into a comprehensive dict

### Patterns
- Confusion matrix: TP (correct+confident), FP (wrong+confident), TN (wrong+uncertain), FN (correct+uncertain)
- Prevention rate = TN/(TN+FP) = fraction of hallucinations caught
- Unnecessary block rate = FN/(FN+TP) = fraction of correct answers blocked
- Empty arrays require explicit dtype handling for bitwise operations

### Gotchas
- Empty labels array causes TypeError on bitwise AND unless explicitly cast to bool dtype
- Run pytest with `uv run python -m pytest` not just `uv run pytest` for proper module resolution
- test_data.py also has the same import issue - tests were already failing before this task

### Functions Implemented
1. confusion_matrix_at_threshold(scores, labels, threshold) -> dict
2. prevention_rate_at_threshold(scores, labels, threshold) -> float  
3. unnecessary_block_rate_at_threshold(scores, labels, threshold) -> float
4. selective_accuracy(direct_correct, cot_correct, abstentions, total) -> float
5. token_efficiency(direct_tokens, cot_tokens, routed_tokens, total) -> dict
6. calibration_curve(scores, labels, n_bins) -> dict
7. compute_all_metrics(scores, labels, threshold) -> dict

## Task 3: Probe Scoring Utilities (probe.py)

### Context
- Implemented 8 functions in src/probe.py for probe scoring and threshold analysis
- Created 27 tests in tests/test_probe.py

### Decisions Made
- Used scipy.special.expit for numerical stability in sigmoid
- JSON and NPZ formats supported for probe weights
- threshold_sweep returns DataFrame with multiple metrics at each threshold

### Patterns
- compute_confidence: g(h_ℓ(x)) = σ(w^T h_ℓ(x) + b) ∈ [0,1]
- AUROC edge case: return float('nan') when all labels same
- Prevention rate = TN/(TN+FP) = fraction of hallucinations caught by low probe score
- Unnecessary block rate = FN/(FN+TP) = fraction of correct answers blocked

### Gotchas
- .is_nan() method doesn't exist on plain float - use np.isnan() instead
- test_threshold_shift: 0.4 < 0.55 threshold, not < threshold (FN=2)
- test_unnecessary_block_rate: FN/(FN+TP) not sum/length
- score_dataset takes last token for 2D activations (sequence, hidden)
