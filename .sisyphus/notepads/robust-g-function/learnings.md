# Post-Hoc Calibration Results

## Task: T5 - Post-Hoc Calibration (Platt scaling + Isotonic regression)

**Date:** 2026-05-09

## Key Findings

### Platt Scaling → KEEP
- ECE reduction: 32.1% (raw 0.1022 → calibrated 0.0694)
- AUROC drop: -0.0004 (essentially NO change,甚至略有提升)
- Variance stable across seeds (Levene's p=0.74)

### Isotonic Regression → INVESTIGATE
- ECE reduction: 57.0% (raw 0.1022 → calibrated 0.0439)
- AUROC drop: 0.0143 (超过0.01阈值，超过了pre-registered的容忍度)
- High variance across seeds (std=0.024 vs Platt std=0.005)
- One seed (seed=42) had AUROC drop to 0.906 — concerning instability

## Why Platt is preferred over Isotonic
1. Platt preserves discrimination (AUROC unchanged)
2. Platt has stable ECE variance across seeds
3. Isotonic is non-parametric — can overfit to calibration set, especially with small calibration sets (n=67)

## Decision
- **Platt scaling: KEEP** (ECE reduction ≥ 30% AND AUROC drop < 0.01 ✓)
- **Isotonic regression: INVESTIGATE** (ECE reduction ≥ 30% ✓ but AUROC drop = 0.014 > 0.01 ✗)

## Notes
- 5-seed protocol with 20/80 calibration/test split
- 335 total held-out samples
- Calibration set per seed: 67 samples
- Test set per seed: 268 samples

## Files Created
- `scripts/posthoc_calibration.py`
- `data/ablation_results/calibration_5seed.json`