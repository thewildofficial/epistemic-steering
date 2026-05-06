"""Probe scoring and uncertainty quantification.

Computes epistemic confidence scores from LLM hidden states using
logistic regression probes trained on the Qwen3.5-4B model.

Core functions:
- load_probe_weights: Load trained probe coefficients
- compute_confidence: g(h_ℓ(x)) = σ(w^T h_ℓ(x) + b) ∈ [0,1]
- score_dataset: Batch confidence scoring
- threshold_sweep: Metric computation across thresholds
"""