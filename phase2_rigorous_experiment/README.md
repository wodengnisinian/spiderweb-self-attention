# Phase 2: Rigorous Multi-Seed Experiment

**Date**: 2026-06-11

## Objective
Add statistical rigor: 5 seeds, RandomBias control, per-class metrics, length-grouped analysis.

## What was done
- Added evaluate_detailed() to train.py (returns preds/labels/seq_lens)
- Added uild_m_random() to data.py (random bias control)
- Rewrote experiment.py: 5 seeds × 5 variants × 10 epochs
- Added Accuracy, Precision, Recall, Macro-F1, Weighted-F1 metrics
- Added length-grouped evaluation: short (< 50), medium (50–65), long (> 65)
- Saved all results as CSV

## Variants
| Variant | Description |
|---------|-------------|
| A: Transformer | Baseline |
| B: +Position | Position bias only |
| C: +Simple Structure | Hierarchy + position |
| D: +SpiderWeb | Full SpiderWeb bias |
| E: +Random Bias | Random matrix with same stats (control) |

## Key result
SpiderWeb 0.8080 ± 0.0265 vs Transformer 0.7870 ± 0.0210 (+2.10 pp, p=0.0061)

## Files
- experiment_results.csv — 25 rows (5 seeds × 5 variants), all metrics
- experiment_results.json — same data in JSON format
