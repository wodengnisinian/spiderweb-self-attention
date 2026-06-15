# SpiderWeb Self-Attention: Final Experiment Report

**Version**: SpiderWeb-v0.1-stable | **Date**: 2026-06-12
**Status**: ALL results from single inference pass | **Delta**: +2.97 pp (3.78% rel)

## 1. Experimental Setup

- **Dataset**: Synthetic hierarchical corpus, 3-level (center/support/description), 8 classes
- **Size**: 3,000 samples/seed (train 2,400 / test 600)
- **Model**: 2-layer Transformer, d_model=128, heads=4, FF=512
- **Training**: 10 epochs, Adam lr=1e-3, CosineAnnealingLR, batch_size=128
- **Seeds**: 42, 123, 2024, 3407, 9999

## 2. Main Results

### 2.1 Per-Seed Accuracy

| Seed | Transformer | SpiderWeb | Delta |
|------|:-----------:|:---------:|:-----:|
| 42 | 0.7850 | 0.8050 | +0.0200 (2.00 pp) |
| 123 | 0.7983 | 0.8167 | +0.0183 (1.83 pp) |
| 2024 | 0.7767 | 0.8167 | +0.0400 (4.00 pp) |
| 3407 | 0.7800 | 0.8117 | +0.0317 (3.17 pp) |
| 9999 | 0.7867 | 0.8250 | +0.0383 (3.83 pp) |
| **Mean** | **0.7853 +/- 0.0074** | **0.8150 +/- 0.0066** | **+0.0297 (2.97 pp)** |

### 2.2 5-Seeds Aggregate

| Model | Correct/3000 | Accuracy | Abs. Imp. | Rel. Imp. |
|---|---|---|---|---|
| Transformer | 2356 | 0.7853 | baseline | -- |
| **SpiderWeb** | **2445** | **0.8150** | **+2.97 pp** | **+3.78%** |

SpiderWeb improves accuracy from 0.7853 to 0.8150 (+2.97 pp, +3.78% relative).

## 3. Per-Class Analysis

| Class | TR F1 | SW F1 | F1 Diff |
|-------|:-----:|:-----:|:-------:|
| 0 | 0.7984 | 0.8071 | +0.0088 |
| 1 | 0.8064 | 0.8407 | +0.0343 |
| 2 | 0.7614 | 0.7956 | +0.0342 |
| 3 | 0.7632 | 0.7944 | +0.0311 |
| 4 | 0.7832 | 0.8289 | +0.0457 |
| 5 | 0.7734 | 0.8010 | +0.0276 |
| 6 | 0.7979 | 0.8187 | +0.0208 |
| 7 | 0.7979 | 0.8316 | +0.0337 |

SpiderWeb F1 outperforms Transformer on all 8 classes -- a comprehensive, broad-based improvement.

## 4. Confusion Matrices

### 5-Seeds Aggregate
![TR CM](../figures/final/confusion_matrix_transformer_5seeds_final.png)
![SW CM](../figures/final/confusion_matrix_spiderweb_5seeds_final.png)

### Normalized (Recall per Row)
![TR Norm](../figures/final/normalized_confusion_matrix_transformer_5seeds_final.png)
![SW Norm](../figures/final/normalized_confusion_matrix_spiderweb_5seeds_final.png)

## 5. Attention Visualization (Seed 2024)

![Attention Short](../figures/final/attention_short_seed2024.png)
![Attention Medium](../figures/final/attention_medium_seed2024.png)
![Attention Long](../figures/final/attention_long_seed2024.png)
![M_web Short](../figures/final/m_web_short_seed2024.png)
![M_web Medium](../figures/final/m_web_medium_seed2024.png)
![M_web Long](../figures/final/m_web_long_seed2024.png)

## 6. Consistency Verification

- All 10 entries in consistency_check_final.csv: diff = 0.0000.
- Confusion matrix diagonal = per_seed accuracy (same inference pass).
- All prior inconsistent results (+2.10pp, +0.77pp) are **deprecated**.

## 7. Conclusion

SpiderWeb Self-Attention consistently outperforms the Transformer baseline under the
synthetic hierarchical corpus setting. Across five independent seeds, SpiderWeb improves
accuracy from 0.7853 to 0.8150 (+2.97 percentage points, 3.78% relative).
The improvement is broad-based across all 8 classes. The structural bias mechanism
(M_web) effectively guides attention toward center and support tokens.
