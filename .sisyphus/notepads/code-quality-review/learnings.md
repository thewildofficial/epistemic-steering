# Code Quality Review — Learnings

## Files Reviewed (10 total)
8 new untracked scripts + 2 recently modified files.

## Unused Imports Found (all minor, no functional impact)
| File | Unused Import(s) | Severity |
|------|-------------------|----------|
| `scripts/layer_sweep_baseline.py` | `calibration_curve` (sklearn.calibration) | Low |
| `scripts/ensemble_probe.py` | `t as t_dist` (scipy.stats), `LogisticRegression` (sklearn), `mean, stdev` (statistics) | Low |
| `scripts/feature_augmentation.py` | `pd` (pandas) | Low |
| `scripts/posthoc_calibration.py` | `CalibratedClassifierCV` (sklearn) | Low |
| `scripts/brier_loss_train.py` | `CalibratedClassifierCV` (sklearn), `levene` (scipy.stats) | Low |
| `scripts/cross_benchmark_eval.py` | `sys` | Low |

## Clean Files (0 issues)
- `scripts/calibration_audit.py`
- `scripts/conformal_prediction.py`
- `scripts/evaluate_heldout.py`
- `src/steering.py`

## AI Slop Check
- No TODO/FIXME/HACK/xxx markers
- No commented-out code detected
- No verbose logging (no logging module usage)
- No excessive docstrings
- No over-abstraction — functions are appropriately scoped
- Variable names are descriptive and domain-appropriate

## Pattern: Copied Import Blocks
The unused imports across multiple files follow a common pattern — import blocks were copied between scripts and not cleaned up. Example: `CalibratedClassifierCV` appears unused in both `posthoc_calibration.py` and `brier_loss_train.py`.
