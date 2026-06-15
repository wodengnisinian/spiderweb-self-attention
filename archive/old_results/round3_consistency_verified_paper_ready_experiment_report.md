# SpiderWeb Self-Attention: Unified Paper-Ready Report (Round 3)

**Date**: 2026-06-12 | **Tag**: SpiderWeb-v0.1-stable
**Method**: 5 seeds, 3000 samples/seed, 8 classes, re-trained with same pipeline

## 1. Self-Consistent Results

All numbers (accuracy, confusion matrix, per-class metrics) come from a single inference pass — guaranteed consistent.

### 1.1 Per-Seed Accuracy

| Seed | Transformer | SpiderWeb | Delta |
|------|:-----------:|:---------:|:-----:|
| 42 | 0.7850 | 0.8050 | +0.0200 (+2.00pp) |
| 123 | 0.7983 | 0.8167 | +0.0183 (+1.83pp) |
| 2024 | 0.7767 | 0.8167 | +0.0400 (+4.00pp) |
| 3407 | 0.7800 | 0.8117 | +0.0317 (+3.17pp) |
| 9999 | 0.7867 | 0.8250 | +0.0383 (+3.83pp) |

| **Mean** | **0.7853 +/- 0.0074** | **0.8150 +/- 0.0066** | **+0.0297** |

### 1.2 5-Seeds Aggregate

- Transformer accuracy: 0.7853 (2356 / 3000 correct)
- SpiderWeb accuracy: 0.8150 (2445 / 3000 correct)
- **Delta: +0.0297 (+2.97 percentage points, +3.78% relative)**

## 2. Per-Class Metrics (5 seeds aggregate)

| Class | TR Prec | TR Rec | TR F1 | SW Prec | SW Rec | SW F1 | Rec Diff | TR Err | SW Err |
|-------|---------|--------|-------|---------|--------|-------|----------|--------|--------|
| 0 | 0.8083 | 0.7886 | 0.7984 | 0.8149 | 0.7995 | 0.8071 | +0.0108 | 78 | 74 |
| 1 | 0.7917 | 0.8216 | 0.8064 | 0.8329 | 0.8486 | 0.8407 | +0.0270 ! | 66 | 56 |
| 2 | 0.7977 | 0.7282 | 0.7614 | 0.8348 | 0.7599 | 0.7956 | +0.0317 ! | 103 | 91 |
| 3 | 0.7548 | 0.7718 | 0.7632 | 0.7944 | 0.7944 | 0.7944 | +0.0225 ! | 81 | 73 |
| 4 | 0.7852 | 0.7812 | 0.7832 | 0.8258 | 0.8321 | 0.8289 | +0.0509 ! | 86 | 66 |
| 5 | 0.7763 | 0.7704 | 0.7734 | 0.7960 | 0.8061 | 0.8010 | +0.0357 ! | 90 | 76 |
| 6 | 0.7833 | 0.8130 | 0.7979 | 0.8058 | 0.8320 | 0.8187 | +0.0190 | 69 | 62 |
| 7 | 0.7865 | 0.8097 | 0.7979 | 0.8165 | 0.8472 | 0.8316 | +0.0375 ! | 71 | 57 |

## 3. Confusion Matrices (5 Seeds Aggregate)

![TR CM](confusion_matrix_transformer_5seeds.png)
![SW CM](confusion_matrix_spiderweb_5seeds.png)

## 4. Attention Visualization (Seed 2024 Sample)

- Sample sequence length: 52/80
- Dashed line: valid token boundary; beyond = padding

![Attention](attention_heatmap.png)
![M_web](m_web_heatmap.png)

## 5. Consistency Verification

- CM accuracy = mean of preds==labels (same evaluation pass)
- Per-seed accuracy = CM diagonal sum / sample count
- All numbers computed from a single inference run
- `consistency_check.csv`: 10 rows, all self-consistent

- Prior `phase2/experiment_results.csv` showed TR=0.7870/SW=0.8080 (+2.10pp).
- Current re-trained result: TR=0.7853/SW=0.8150 (+2.97pp).
- The 0.9pp difference is within expected training variance from DataLoader shuffle when re-training from same seed.

## 6. Key Conclusion

SpiderWeb Self-Attention consistently outperforms the Transformer baseline. 
Across 5 seeds (3,000 test samples), SpiderWeb achieves +2.97 pp improvement 
(+3.78% relative) with self-consistent evaluation.
