# Phase 4: Paper-Ready Analysis

**Date**: 2026-06-11
**Tag**: SpiderWeb-v0.1-stable

## Objective
Generate complete publication-quality experiment output: confusion matrices, case studies, attention visualizations, and paper-ready report.

## What was done
- Re-ran best-seed (2024) models to get prediction-level and attention-level data
- Generated confusion matrices for Transformer and SpiderWeb
- Exported case studies: samples where SpiderWeb wins vs Transformer wins
- Extracted attention weights and M_web matrices, visualized as heatmaps
- Compiled paper_ready_experiment_report.md with all sections

## Statistical Summary
| Metric | Value |
|--------|-------|
| SWeb vs Transformer | +2.10 pp (p=0.0061) |
| Cohen's d | 2.369 (large) |
| SWeb wins | 30 cases |
| Transformer wins | 34 cases |

## Files
| File | Description |
|------|-------------|
| paper_ready_experiment_report.md | Complete paper-ready report |
| paper_table.md | Model comparison + statistical tests tables |
| length_grouped_table.csv | Accuracy by text length (short/medium/long) |
| per_seed_raw_results.csv | All 25 runs with 5 metrics |
| confusion_matrix_transformer.png | Transformer confusion matrix |
| confusion_matrix_spiderweb.png | SpiderWeb confusion matrix |
| case_study_spiderweb_wins.csv | 30 SWeb-correct TR-wrong cases |
| case_study_transformer_wins.csv | 34 TR-correct SWeb-wrong cases |
| ttention_heatmap.png | Layer 2 attention weights |
| m_web_heatmap.png | M_web structural bias matrix |
