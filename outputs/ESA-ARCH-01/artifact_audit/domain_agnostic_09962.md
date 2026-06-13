# ESA-ARCH-01 Artifact Audit: Domain-Agnostic AUROC 0.9962

**Date:** 2026-06-13
**Machine:** naam-machine4000pro
**Artifact:** Mean domain-agnostic AUROC = 0.9962 reported for max-pooled trajectory features
**Status:** ARTIFACT (evaluation leakage)

## Summary

The reported 0.9962 mean domain-agnostic AUROC is **not evidence of a universal correctness probe**. The protocol that produced it trains on all four domains pooled and then tests on each domain whose samples are already present in the training pool. When the evaluation is changed to a proper leave-one-domain-out split, mean AUROC collapses to **~0.48**, near chance and consistent with the poor cross-domain transfer mean of **0.5937**. I therefore classify 0.9962 as an **artifact of test-set contamination**, not a real signal.

---

## 1. Exact Source of 0.9962

| Item | Value |
|---|---|
| File | `/home/guest/epistemic-steering/scripts/cross_domain_test.py` |
| Function | `evaluate_domain_agnostic(method)` |
| Lines | ~166-197 (as shown by `sed -n 140,220p`) |
| Command | `python scripts/cross_domain_test.py` producing `outputs/trajectory_observer/cross_domain_results.json` |
| Reported value | `"mean": 0.9962` under `domain_agnostic.max_pooled` |

The function performs the following steps:
1. Loads max-pooled features for each of the four domains (`arc`, `HumanEval`, `math`, `triviaqa`).
2. Concatenates them into a single pooled matrix `all_X` and label vector `all_y`.
3. For every domain `test_domain` in turn, trains a `LogisticRegressionCV` probe on the pooled data and evaluates it on `features[test_domain]`.
4. Averages the four per-domain AUROCs (0.9872, 0.9997, 0.9980, 0.9999) to obtain **0.9962**.

This procedure is recorded in:
- `outputs/trajectory_observer/cross_domain_results.json` (`domain_agnostic.max_pooled.mean = 0.9962`)
- `outputs/trajectory_observer/cross_domain_matrix.txt` (`Mean domain-agnostic AUROC: 0.9962`)
- `outputs/trajectory_observer/comparison_report.md` (`Domain-agnostic pooled AUROC: **0.9962**`)

The metadata file `data/activations_allpos/cost_log_allpos.json` contains only benchmark run-time/accuracy metadata; it does **not** explain or justify the 0.9962 AUROC computation.

---

## 2. Contamination Hypotheses and Evidence

### Hypothesis A: Test samples are included in the training pool (confirmed)

`evaluate_domain_agnostic` builds `all_X`/`all_y` by concatenating **all four domains**, including the target domain it later evaluates on. For `all->arc`, the 200 arc test samples are a strict subset of the 764-sample training pool. The same holds for every domain:

| Test domain | Train size | Test size | Overlap with train | Overlap/test |
|---|---|---|---|---|
| arc | 764 | 200 | 200 | 100% |
| HumanEval | 764 | 164 | 164 | 100% |
| math | 764 | 200 | 200 | 100% |
| triviaqa | 764 | 200 | 200 | 100% |

This is not a sample-index overlap between distinct examples; the **same feature vectors and labels** are used for both training and evaluation. The cached pickle (`outputs/ESA-ARCH-01/artifact_audit/maxpooled_features.pkl`, 54.8 MB) contains exactly the same rows per domain, and exact-row duplicate checking showed each row matches only itself in the pooled set.

### Hypothesis B: Duplicate examples across domains causing hidden leakage (ruled out)

I checked the cached pickle for exact feature-vector duplicates across domains. Each row in each domain matches exactly one row in the concatenated pool, so there are no duplicate samples shared between domains. Contamination is purely protocol-level, not data-level.

### Hypothesis C: Class-imbalance shortcut (partial factor)

Domain base rates vary widely:
- arc: 90.5% correct
- HumanEval: 32.3% correct
- math: 61.5% correct
- triviaqa: 38.0% correct

The reported accuracies in the original run (e.g., `all->arc` accuracy 0.905) match the arc base rate, suggesting the probe can trivially exploit prevalence. However, AUROC is prevalence-independent, and the plain-LR reproduction still reaches AUROC = 1.0000 under the same leaky protocol, so the dominant issue is leakage, not imbalance.

---

## 3. Reproduction Attempt Outcome

I used the cached pickle to run two reproductions on naam-machine4000pro, bypassing `cross_domain_test.py` entirely.

### 3.1 Original protocol reproduced from pickle

| Pair | AUROC | Accuracy |
|---|---|---|
| all->arc | 1.0000 | 1.0000 |
| all->HumanEval | 1.0000 | 1.0000 |
| all->math | 1.0000 | 1.0000 |
| all->triviaqa | 1.0000 | 1.0000 |
| **Mean** | **1.0000** | **1.0000** |

Using `LogisticRegression(solver='liblinear')` on the cached max-pooled features, the original protocol gives **perfect AUROC**. The original 0.9962 used `LogisticRegressionCV`, which adds mild regularization; the difference is minor and the behavior is identical: testing on training data yields near-perfect scores.

### 3.2 Proper leave-one-domain-out protocol

| Train domains | Test domain | AUROC | Accuracy |
|---|---|---|---|
| HumanEval+math+triviaqa | arc | 0.4196 | 0.0950 |
| arc+math+triviaqa | HumanEval | 0.4748 | 0.6159 |
| arc+HumanEval+triviaqa | math | 0.5335 | 0.3850 |
| arc+HumanEval+math | triviaqa | 0.4828 | 0.3850 |
| **Mean** | | **0.4777** | - |

When the test domain is truly held out, performance drops to approximately chance, far below the cross-domain transfer mean of 0.5937. This confirms the universal probe cannot reliably generalize to an unseen domain.

Raw results saved on the remote host at `/tmp/repro_9962_results.json`.

---

## 4. Verdict

**Verdict: ARTIFACT**

The 0.9962 domain-agnostic AUROC is produced by evaluating the probe on data that was included in its training set. It does not demonstrate a universal correctness signal. Under a correct leave-one-domain-out evaluation, the same cached features yield a mean AUROC of ~0.48, i.e. near chance.

---

## 5. Recommendations

1. **Retire 0.9962** from any claims about universal probe performance.
2. **Replace the domain-agnostic metric** in `scripts/cross_domain_test.py` with a leave-one-domain-out protocol if the goal is to measure generalization across ARC, HumanEval, MATH, and TriviaQA.
3. **Clarify the comparison_report.md** conclusion: the only meaningful generalization numbers are the cross-domain transfer mean (0.5937) and the per-pair matrix, not the domain-agnostic pooled score.
4. If stronger universal generalization is desired, consider domain-invariant training or domain-adversarial methods, but do not claim it from the current 0.9962 figure.

---

## Audit Log

- Read `data/activations_allpos/cost_log_allpos.json`: benchmark runtime/accuracy metadata only.
- Grepped repository for `0.9962`, `9962`, `domain_agnostic`, `all_domain`.
- Read `outputs/trajectory_observer/cross_domain_results.json` and `outputs/trajectory_observer/comparison_report.md`.
- Loaded cached pickle `outputs/ESA-ARCH-01/artifact_audit/maxpooled_features.pkl` (54.8 MB, 764 samples, 17,920 dims).
- Checked exact duplicate rows across domains: none.
- Reproduced original protocol: AUROC = 1.0000 per domain.
- Ran leave-one-domain-out protocol: mean AUROC = 0.4777.
- Report written by autonomous audit agent to `outputs/ESA-ARCH-01/artifact_audit/domain_agnostic_09962.md`.
