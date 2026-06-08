## 2026-06-08: hewitt_liang_controls.py

**File created:** `scripts/hewitt_liang_controls.py`  
**Status:** ✅ Mock data passes all 3 controls

### Key design decisions:
- **3 controls implemented**: random_labels, position_only, difficulty_only
- **Spurious threshold**: AUROC > 0.60 triggers "SPURIOUS" conclusion
- **Seeds**: shuffle uses seed 999, noise uses 777, synth uses 555 — all distinct from training seeds [42, 123, 456, 789, 2024]
- **Mock mode** uses 20-dim features (MOCK_DIM) instead of 2560 to avoid high-D overfitting where LogisticRegressionCV separates random noise
- **Difficulty-only control**: trains on real benchmark, evaluates on synthetic math problems (2×2 vs 10×10 matrix multiplies) with difficulty-correlated hidden state structure
- **Output**: saves to `results/hewitt_liang_controls.json`

### Mock test results (all PASS):
- Random labels AUROC: 0.543 ± 0.000
- Position-only AUROC: 0.523 ± 0.000
- Difficulty-only AUROC: 0.504
- Overall: ALL CONTROLS PASSED