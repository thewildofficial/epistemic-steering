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

## Task 7: Figures + Visualization Pipeline

### Context
- Implemented src/plotting.py with 6 reusable plotting functions
- Created scripts/generate_figures.py to produce 5 publication-quality figures
- Generated 9 PNG (300 dpi) and 9 PDF files in figures/ directory

### Decisions Made
- Used Seaborn whitegrid theme with colorblind palette for consistency
- Font sizes ≥ 12pt for readability
- All figures saved as both PNG (300 dpi) and PDF
- ROC curves approximated from threshold sweep data (no raw predictions available)
- Optimal threshold defined as maximizing F1 score

### Patterns
- _setup_style() applies consistent Seaborn theme
- _save_figure() handles both PNG and PDF output
- Each figure function follows same signature: data params + title + save_path
- Threshold sweep data contains prevention_rate and unnecessary_block_rate
- FPR ≈ 1 - prevention_rate, TPR ≈ 1 - unnecessary_block_rate

### Gotchas
- Matplotlib backend issues on macOS - use 'Agg' if needed
- Path objects need conversion to string for save_path parameter
- LSP import errors expected (virtual environment not recognized)
- ROC curve approximation may not be perfect but sufficient for visualization
- Calibration curve bin_counts parameter optional but improves visualization

### Functions Implemented
1. plot_auroc_curve(fpr, tpr, auroc, title, save_path)
2. plot_confusion_matrix_heatmap(cm, labels, title, save_path)
3. plot_threshold_tradeoff(thresholds, prevention, unnecessary_block, optimal_threshold, save_path)
4. plot_calibration_curve(bin_centers, observed_accuracy, title, save_path, bin_counts)
5. plot_dataset_comparison(mmlu_metrics, gsm8k_metrics, save_path)
6. plot_prevention_rate_curve(thresholds, prevention_rates, optimal_threshold, save_path, dataset_name)

### Figures Generated
1. Figure 1: AUROC curves for MMLU (0.827) and GSM8K (0.994, overfit)
2. Figure 2: Threshold tradeoff curves with optimal threshold markers
3. Figure 3: Calibration curves (reliability diagrams) for both datasets
4. Figure 4: Prevention rate vs threshold with optimal threshold marked
5. Figure 5: MMLU vs GSM8K comparison bar chart

## Task 8: Generation-Time Hidden State Extraction on Modal

### Context
- Created scripts/extract_gen_time_data.py to run on Modal T4 GPU
- Created scripts/download_gen_time_data.py to pull results locally
- Qwen3.5-4B already cached on Modal volume `epistemic-model-cache` at /vol/model
- 656 questions with prompts in /vol/results/probe_extract_results.jsonl
- Layer 30 is probe layer (same as prefill extraction)

### Decisions Made
- Used forward hooks on `model.model.layers[30]` to capture hidden states during `model.generate()`
- Skipped the first hook invocation (prefill) to capture only generation-time states
- Sampled every 5th token to stay within budget and keep data volume manageable
- Used `pickle` for saving results (handles nested list-of-arrays structure better than np.savez)
- Set timeout to 6 hours (21_600 sec) — conservative for 656 questions on T4
- Saved intermediate checkpoints every 50 questions for crash resilience
- Used float16 for memory efficiency on T4 (16 GB VRAM)

### Patterns
- Hook closure with mutable `step_counter = [0]` to distinguish prefill from generation steps
- `output[0][:, -1, :]` extracts last-token hidden state at each forward pass
- `output_ids[0][input_ids.shape[1]:]` decodes only newly generated tokens
- Cost tracking: `elapsed * 0.000164` (T4 rate $0.000164/sec ≈ $0.59/hr)
- Download script follows same pattern as verify_insamp.py: `modal.Volume.from_name()` → `vol.listdir()` → `vol.read_file()`

### Gotchas
- `np.savez_compressed` cannot save arbitrary nested dicts with lists of arrays — pickle is safer
- T4 timeout default is too short for 656 questions; must set explicitly (3600s = 1hr is insufficient)
- `trust_remote_code=True` required for Qwen3.5-4B tokenizer and model loading
- `pad_token_id` must be set explicitly if tokenizer lacks a pad token
- Questions with model_answer="?" should still be generated (no special skipping logic needed)
- Actual runtime likely 2-3 hours (not 5.5), keeping cost well under $8 budget (~$1.50-$3.25)


## 2026-05-06: scripts/compare_methods.py

### Implementation
- Script consolidates in-sample verification results into comparison tables with bootstrap CIs
- Computes 4 methods: Always Direct, Always CoT, Random Routing, Prefill Probe (Ours)
- Uses `token_efficiency()` from `src/evaluate.py` as base, but overrides `savings_vs_always_cot` because the stock function compares against actual CoT tokens used, not the always-CoT baseline
- Bootstrap CI: 1000 resamples with seed=42, 95% CI

### Key numbers (MMLU, t=0.5, in-sample):
| Method             | Accuracy | 95% CI           | Tok/Q | Abstention |
|--------------------|----------|------------------|-------|------------|
| Always Direct      | 0.557    | [0.509, 0.601]   | 8.0   | 0%         |
| Always CoT         | 0.597    | [0.548, 0.640]   | 120.0 | 0%         |
| Random Routing     | 0.385    | [0.340, 0.428]   | 42.7  | 33%        |
| Prefill Probe      | 0.904    | [0.875, 0.930]   | 63.3  | 49%        |

### Statistical tests:
- Prefill vs Direct: Δ=+0.347, CI [0.305, 0.393] — significant
- Prefill conservative (80% CoT) vs Direct: Δ=+0.276, CI [0.237, 0.318] — significant
- GSM8K: tests omitted due to overfit (0.994 AUROC, 7/200 positives)

### Caveats:
- CoT accuracy for Always CoT and Random Routing is estimated (+4pp MMLU, +10pp GSM8K)
- Held-out evaluation on Modal GPU will replace estimates
- GSM8K probe is severely overfit — do not claim GSM8K steering works
- GSM8K selective accuracy (0.995) is meaningless due to 3.5% base accuracy

### Output files:
- data/comparison_results.json — per-method metrics with bootstrap CIs
- figures/fig6_accuracy_comparison.{png,pdf} — bar chart
- figures/fig7_selective_accuracy_vs_abstention.{png,pdf} — scatter with threshold sweep
- figures/fig8_token_efficiency.{png,pdf} — bubble plot

### Watch out for:
- Figure fonts: ⚠ (U+26A0) may not render on all systems. Used text "WARNING:" instead.
- The `token_efficiency()` function's `savings_vs_cot` compares against actual CoT usage, not always-CoT. Compute `savings_vs_always_cot` manually.

## 2026-05-10: Generation-Time Probe V3 Full Run

### Result
- Full run trained per-position logistic-regression probes on 197 GSM8K generation traces.
- Each sample at position `t` is the layer-25 hidden state at generated token `t`, labeled by final-answer correctness.
- Optimal position: 3158.
- Peak test AUROC: 0.9129.
- Brier score at peak: 0.1858.
- ECE at peak: 0.2387.
- Scenario: B (late-jump).
- Calibration verdict: NEEDS FIXING (ECE >= 0.10).

### Interpretation
- The high AUROC indicates strong late-stage separability: hidden states deep in long reasoning traces contain information about whether the final answer will be correct.
- The signal is not an early monotonic confidence signal. It peaks very late, so it supports generation-time monitoring but is less useful for cheap early routing unless earlier positions also perform well.
- Poor ECE means the raw logistic-regression probabilities should not be treated as calibrated probabilities. Use the probe as a ranking signal, or calibrate with Platt/isotonic calibration before threshold-based steering.
- GSM8K remains high-risk for overclaiming because of class imbalance and shrinking sample counts at late token positions. Validate on held-out generations before making a strong claim.

### Paper-safe wording
Generation-time layer-25 hidden states on GSM8K show a strong late-stage correctness signal, peaking at AUROC 0.9129 around token 3158. However, calibration is poor (ECE 0.2387), so the result supports separability/ranking rather than directly reliable confidence estimates.

## Task 1: Calibration Audit (scripts/calibration_audit.py)

### Context
- Auditing the existing probe g(h_30(x)) = σ(w^T h_30(x) + b) for calibration quality
- 5-seed cross-validation protocol on 656 training samples
- Evaluated on 335 held-out samples from data/heldout_eval/final_summary.json
- Project uses `uv run python` for execution (pyproject.toml dependencies)

### Key Results
| Metric | Held-Out | 5-Seed CV (mean ± std) |
|--------|----------|------------------------|
| ECE | 0.1011 | 0.3140 ± 0.0180 |
| Brier | 0.0760 | 0.3116 ± 0.0182 |
| AUROC | 0.9678 | 0.8292 ± 0.0228 |
| AUPRC | 0.9081 | 0.7324 ± 0.0541 |

### Decision
**Calibration NEEDS FIXING (ECE = 0.1011 >= 0.05)**

The probe is poorly calibrated on held-out data: when it predicts 50% confidence, actual accuracy is only ~40%. The training-free probe (mean-diff direction) also shows poor in-sample calibration (ECE ~0.31), indicating the mean-diff direction approach doesn't produce well-calibrated probabilities.

### Patterns
- Levene's test returns NaN when values have near-zero variance or when scipy produces nan (edge case in variance homogeneity testing)
- Calibration curve uses sklearn.calibration_curve with strategy='uniform' and 10 bins
- Brier score computed via sklearn.metrics.brier_score_loss
- ECE computed manually as sum_i (count_i/n) * |acc_i - conf_i| with equal-width bins

### Implementation Notes
- Uses StratifiedKFold(5) with shuffle=True, random_state=None (different each run)
- Training-free probe: direction = mean(correct) - mean(incorrect), normalized, then expit(projection)
- Held-out probe scores loaded from data/heldout_eval/final_summary.json (pre-computed by existing probe)
- Output files:
  - data/ablation_results/baseline_5seed.json — full metrics + per-sample predictions
  - .sisyphus/evidence/task-1-calibration-curve.png — reliability diagram

### Gotchas
- scipy.stats.levene can produce nan when variance is near-zero or in edge cases — handle with np.std check before calling
- matplotlib backend must be set to 'Agg' on headless environments
- activation file naming: {qid}__layer_{layer}.npy or q{qid}_layer_{layer}.npy — try both patterns
- MMLU model_answer='?' should be treated as incorrect (mark correct=False)
