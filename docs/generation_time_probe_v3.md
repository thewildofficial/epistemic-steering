# Generation-Time Probe V3 Results

## Summary

Probe V3 trains one logistic-regression probe per generated-token position. For a fixed position `t`, each sample is the Qwen3.5-4B layer-25 hidden state at generated token `t`, labeled by whether the final GSM8K answer for that generation was correct.

Latest full run:

| Metric | Value |
|---|---:|
| Questions | 197 |
| Optimal token position | 3158 |
| Test AUROC at optimum | 0.9129 |
| Brier score at optimum | 0.1858 |
| ECE at optimum | 0.2387 |
| Scenario | B (late-jump) |
| Calibration verdict | Needs fixing |

## Interpretation

The high AUROC means late-generation hidden states contain a strong separability signal for final-answer correctness. At token position 3158, the probe can rank correct generations above incorrect generations with AUROC 0.9129. This is meaningful evidence that the model's internal state becomes more diagnostic during long reasoning traces.

The result is a late-jump pattern, not an early monotonic signal. The most predictive position occurs deep in the generated chain, after many shorter generations have already ended. This suggests the signal is strongest in long-running reasoning trajectories, especially cases where the model is still generating thousands of tokens. It is less useful as an early-exit router unless earlier positions also show acceptable AUROC.

The calibration result is poor. ECE 0.2387 means the raw logistic-regression probabilities should not be interpreted as reliable probabilities of correctness. The probe is useful as a ranking signal, but its scores need calibration before being used as thresholds for abstention or continuation decisions.

## Practical Implications

- Use AUROC to claim separability, not calibrated confidence.
- Do not present the raw V3 probability as "the model is 80% likely correct."
- For steering, either calibrate the scores with Platt/isotonic calibration on held-out data or use rank/quantile thresholds.
- The best position at 3158 is too late for cheap intervention in many generations. It supports generation-time monitoring, but not necessarily efficient early stopping.
- Because GSM8K is class-imbalanced and many late positions have fewer samples, validate the late-jump result with held-out generations before making a strong paper claim.

## Paper-Safe Claim

Generation-time layer-25 hidden states on GSM8K show a strong late-stage correctness signal, with peak cross-validated AUROC 0.9129 at token position 3158. However, the probe is poorly calibrated (ECE 0.2387), so the result supports ranking/separability rather than directly usable probability estimates.

