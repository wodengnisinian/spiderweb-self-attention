# SpiderWeb Self-Attention: Corrected Paper-Ready Report (Round 2)

**Date**: 2026-06-11 | **Data**: 5 seeds, 3,000 samples/seed, 8 classes

## 1. 5-Seeds Aggregate Results

- Transformer accuracy: 0.7853 (2356 / 3000)
- SpiderWeb accuracy: 0.7930 (2379 / 3000)
- Delta: **+0.0077** (+0.98%)

## 2. Per-Class Metrics (5 seeds aggregated)

| Class | TR Prec | TR Rec | TR F1 | SW Prec | SW Rec | SW F1 | Rec Diff |
|-------|---------|--------|-------|---------|--------|-------|----------|
| 0 | 0.7989 | 0.7642 | 0.7812 | 0.7832 | 0.8320 | 0.8068 | +0.0678 ! |
| 1 | 0.7624 | 0.8324 | 0.7959 | 0.7698 | 0.8405 | 0.8036 | +0.0081 |
| 2 | 0.8029 | 0.7309 | 0.7652 | 0.8052 | 0.7309 | 0.7663 | +0.0000 |
| 3 | 0.7626 | 0.7690 | 0.7658 | 0.8250 | 0.7437 | 0.7822 | -0.0254 ! |
| 4 | 0.8068 | 0.7863 | 0.7964 | 0.8107 | 0.8066 | 0.8087 | +0.0204 ! |
| 5 | 0.8005 | 0.7883 | 0.7943 | 0.7906 | 0.7704 | 0.7804 | -0.0179 |
| 6 | 0.7923 | 0.7859 | 0.7891 | 0.8085 | 0.8238 | 0.8161 | +0.0379 ! |
| 7 | 0.7605 | 0.8257 | 0.7918 | 0.7596 | 0.7962 | 0.7775 | -0.0295 ! |

## 3. Class 2/4/7 Recall Analysis

| Class | TR Recall | SW Recall | Delta | Notes |
|-------|-----------|-----------|-------|-------|
| 2 | 0.7309 | 0.7309 | +0.0000 | Variance across seeds (s42:0.58/0.70 s123:0.70/0.67 s2024:0.74/0.68 s3407:0.82/0.83 s9999:0.82/0.79) |
| 4 | 0.7863 | 0.8066 | +0.0204 | Variance across seeds (s42:0.78/0.88 s123:0.74/0.78 s2024:0.88/0.81 s3407:0.83/0.83 s9999:0.72/0.73) |
| 7 | 0.8257 | 0.7962 | -0.0295 | Variance across seeds (s42:0.89/0.81 s123:0.77/0.84 s2024:0.87/0.78 s3407:0.82/0.75 s9999:0.78/0.81) |

## 4. Confusion Matrices

### 4.1 Single-Seed (Seed 2024, the best SpiderWeb seed)

These are from *one seed only* for visual inspection. Aggregate below.

![TR Seed 2024](confusion_matrix_transformer_seed2024.png)
![SW Seed 2024](confusion_matrix_spiderweb_seed2024.png)

### 4.2 5-Seeds Sum

Sum of confusion matrices across all 5 seeds (3,000 total test samples).

![TR 5-seeds Sum](confusion_matrix_transformer_5seeds_sum.png)
![SW 5-seeds Sum](confusion_matrix_spiderweb_5seeds_sum.png)

### 4.3 5-Seeds Mean

Average confusion matrix per seed.

![TR 5-seeds Mean](confusion_matrix_transformer_5seeds_mean.png)
![SW 5-seeds Mean](confusion_matrix_spiderweb_5seeds_mean.png)

## 5. Attention Visualization (Sample, Seed 2024)

- Sample sequence length: 52/80 (positions >= 52 are padding)
- Dashed line marks valid sequence length boundary

![Attention](attention_heatmap.png)
![M_web](m_web_heatmap.png)

## 6. Key Corrections (Round 2)

1. Confusion matrices now use 5-seeds aggregate, not single best-seed
2. Chart titles include seed, checkpoint, and split information
3. Both sum and mean versions provided
4. Heatmaps annotated with valid sequence length boundary
5. M_web verified: positions 55+ are valid data for ~27% of samples
6. Per-class recall analysis for degradation classes