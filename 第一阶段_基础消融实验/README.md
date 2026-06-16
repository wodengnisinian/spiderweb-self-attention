# Phase 1: Basic Ablation Study & Lambda Scan

**Date**: 2026-06-11

## Objective
Build the SpiderWeb Self-Attention model from scratch and run initial ablation study.

## What was done
- Implemented model.py (SpiderWeb attention + 4 bias modes)
- Implemented data.py (3-level hierarchical synthetic data)
- Implemented 	rain.py (training pipeline)
- Ran ablation study: 4 variants (A/B/C/D) × 3 repeats × 20 epochs
- Ran lambda scan: 6 lambda values × 3 repeats × 20 epochs

## Variants
| Variant | Description |
|---------|-------------|
| A: Pure Transformer | Baseline, no bias |
| B: +Position | Segment position bias only |
| C: +Simple Structure | Hierarchy + position bias |
| D: +SpiderWeb | Full: center enhancement + hierarchy + position |

## Lambda values scanned
[0.0, 0.1, 0.3, 0.5, 0.7, 1.0]

## Key result
SpiderWeb (D) best accuracy 0.8011 vs Transformer baseline 0.7989, lambda=0.7 optimal.

## Files
- blation_results.json — raw ablation data (12 runs)
- lambda_results.json — raw lambda scan data (18 runs)
- blation_study.png — accuracy curves + bar chart
- lambda_scan.png — lambda scan curves + bar chart
- experiment_report.md — summary report
