# Phase 3: Statistical Analysis & Paper-Ready Tables

**Date**: 2026-06-11

## Objective
Supplement experiments with statistical tests and publication-quality output.

## What was done
- Paired t-tests: SpiderWeb vs Transformer, RandomBias, SimpleStructure
- Wilcoxon signed-rank tests
- 95% confidence intervals
- Paper-formatted comparison table (mean ± std, absolute/relative improvement, p-values)
- Per-seed raw results export
- Corrected chart labels to "pp" (percentage points) + relative improvement

## Statistical results (paired t-test)
| Comparison | Mean Diff | t | p |
|---|---|---|---|
| SpiderWeb vs Transformer | +2.10 pp | 5.30 | 0.0061 |
| SpiderWeb vs RandomBias | +2.67 pp | 3.03 | 0.0388 |
| SpiderWeb vs SimpleStruct | +0.80 pp | 4.50 | 0.0109 |

## Files
- per_seed_raw_results.csv — all 25 runs with 5 metrics
- paper_table.md — publication-ready comparison table
- statistical_tests.md — full test results table
- statistical_report.md — complete analysis report
- ccuracy_bars.png — accuracy bar chart with pp + rel% labels
- 1_bars.png — Macro-F1 bar chart with pp + rel% labels
- length_grouped.png — accuracy by text length (short/medium/long)
