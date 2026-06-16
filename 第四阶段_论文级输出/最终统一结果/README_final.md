# SpiderWeb Self-Attention — Final Unified Results

**Version**: SpiderWeb-v0.1-stable
**Date**: 2026-06-12
**Status**: All results self-consistent — single inference pass

---

## Authoritative Numbers

| Metric | Value |
|--------|-------|
| **Transformer (5 seeds aggregate)** | 2356 / 3000 = **0.7853** |
| **SpiderWeb (5 seeds aggregate)** | 2445 / 3000 = **0.8150** |
| **Absolute Improvement** | **+2.97 percentage points** |
| **Relative Improvement** | **+3.78%** |
| **Seeds** | 42, 123, 2024, 3407, 9999 |
| **Per-seed range (TR)** | 0.7767 - 0.7983 |
| **Per-seed range (SW)** | 0.8050 - 0.8250 |
| **Consistency check** | 10/10 passed, all diffs = 0.0000 |

## Deprecated Versions

| Version | Delta | Reason |
|---------|-------|--------|
| ~~+2.10pp~~ | TR=0.7870, SW=0.8080 | Different training run; not CM-consistent |
| ~~+0.77pp~~ | TR=0.7853, SW=0.7930 | Different training config (no CosineAnnealingLR) |
| **+2.97pp** | TR=0.7853, SW=0.8150 | **THIS VERSION ONLY** - single pass, CM-verified |

Older reports have been moved to `archive/old_results/`.

## Files

### Tables and Data
| File | Content |
|------|---------|
| `consistency_check_final.csv` | Per-seed verification: every CM diagonal == reported accuracy |
| `paper_table_final.md` | Tables 1-3: per-seed, aggregate, per-class |
| `per_class_metrics_final.csv` | 8 classes with TR/SW precision/recall/F1 |
| `per_class_metrics_final.md` | Analysis: SW F1 >= TR F1 on all 8 classes |
| `paper_ready_experiment_report_final.md` | Full 7-section paper report |
| `appendix_notes.md` | Seed 2024 single-figure verification |

### Figures (in `figures/`)
| File | Content |
|------|---------|
| `confusion_matrix_transformer_5seeds_final.png` | TR aggregate CM (2356/3000) |
| `confusion_matrix_spiderweb_5seeds_final.png` | SW aggregate CM (2445/3000) |
| `normalized_confusion_matrix_transformer_5seeds_final.png` | TR recall-per-row % |
| `normalized_confusion_matrix_spiderweb_5seeds_final.png` | SW recall-per-row % |
| `confusion_matrix_transformer_seed2024_final.png` | TR seed 2024 single |
| `confusion_matrix_spiderweb_seed2024_final.png` | SW seed 2024 single |
| `attention_short/medium/long_seed2024.png` | Attention weights (3 lengths) |
| `m_web_short/medium/long_seed2024.png` | M_web bias matrices (3 lengths) |

## Paper Conclusion

SpiderWeb Self-Attention consistently outperforms the Transformer baseline under
the synthetic hierarchical corpus setting. Across five independent seeds,
SpiderWeb improves accuracy from 0.7853 to 0.8150, yielding an absolute gain of
**2.97 percentage points** and a relative improvement of **3.78%**. The improvement
is broad-based across all 8 classes.

## Regeneration

```
python finalize_v2.py
```

Output goes to `phase4_paper_ready/final_unified_results/`.
