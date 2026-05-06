
## F3 Manual QA — 2026-05-06

**Code Imports FAIL**: `steering.py:11` uses `from probe import compute_confidence` (bare import) instead of `from src.probe import compute_confidence`. User's exact import check command fails from project root. Fix: use package-relative import or ensure `src/` is on sys.path in `__init__.py`.

**All other checks passed**:
- Tests: 146 passed, 1 skipped
- Verification: MMLU AUROC 0.8274
- Steering demo: routing output correct
- Paper: compiles to 12-page PDF
- Figures: 24 files (12 PNG + 12 PDF)
