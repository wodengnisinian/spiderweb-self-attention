# SpiderWeb Self-Attention: Paper-Ready Experiment Report

**Version**: SpiderWeb-v0.1-stable | **Date**: 2026-06-11
**Data**: Synthetic hierarchical corpus, 3,000 samples, 8 classes

## 1. Experimental Setup

### 1.1 Dataset
- 3-level hierarchical synthetic corpus: center (L0), support (L1), description (L2)
- 3,000 samples, 8 classes, train/test = 80%/20%
- Length groups: Short (<50), Medium (50-65), Long (>65)

### 1.2 Models
- A: Transformer (baseline) / B: +Position / C: +Simple Structure / D: +SpiderWeb / E: +RandomBias (control)
- d_model=128, heads=4, layers=2, FF=512, epochs=10, Adam lr=1e-3, 5 seeds

## 2. Results

### 2.1 Main Comparison

| Model | Accuracy | Macro-F1 | Abs. Imp. (pp) | Rel. Imp. (%) |
|---|---|---|---|---|
| A: Transformer | 0.7870 +/- 0.0210 | 0.7860 +/- 0.0207 | -- | -- |
| B: +Position | 0.7967 +/- 0.0234 | 0.7959 +/- 0.0237 | +0.97 | +1.23 |
| C: +Simple Struct. | 0.8000 +/- 0.0233 | 0.7993 +/- 0.0235 | +1.30 | +1.65 |
| D: +SpiderWeb | 0.8080 +/- 0.0265 | 0.8070 +/- 0.0272 | +2.10 | +2.67 |
| E: +Random Bias | 0.7813 +/- 0.0218 | 0.7797 +/- 0.0218 | -0.57 | -0.72 |

**SWeb outperforms Transformer by +2.10 pp (relative +2.67%). Cohen's d=2.369.**

### 2.2 Statistical Tests

| Comparison | Metric | Diff (pp) | 95% CI (pp) | t | p | Cohen's d | Wilcoxon p |
|---|---|---|---|---|---|---|---|
| SWeb vs Transformer | accuracy | +2.10 | [+1.00,+3.20] | 5.2963 | 0.0061 | 2.369 | 0.0625 |
| SWeb vs Transformer | macro_f1 | +2.11 | [+0.93,+3.28] | 4.9891 | 0.0075 | 2.231 | 0.0625 |
| SWeb vs RandomBias | accuracy | +2.67 | [+0.22,+5.11] | 3.0291 | 0.0388 | 1.355 | 0.1250 |
| SWeb vs RandomBias | macro_f1 | +2.73 | [+0.25,+5.21] | 3.0561 | 0.0378 | 1.367 | 0.0625 |
| SWeb vs SimpleStruct | accuracy | +0.80 | [+0.31,+1.29] | 4.4961 | 0.0109 | 2.011 | 0.0625 |
| SWeb vs SimpleStruct | macro_f1 | +0.77 | [+0.22,+1.32] | 3.8806 | 0.0178 | 1.735 | 0.0625 |

![Accuracy](accuracy_bars.png)
![F1](f1_bars.png)
![Length](length_grouped.png)
![Transformer CM](confusion_matrix_transformer.png)
![SWeb CM](confusion_matrix_spiderweb.png)
![Attention](attention_heatmap.png)
![M_web](m_web_heatmap.png)

## 3. Conclusion

SpiderWeb Self-Attention achieves statistically significant improvement of +2.10 pp (relative +2.67%) over pure Transformer (t=5.30, p=0.0061, d=2.369). RandomBias underperforms baseline, confirming structural information drives improvement. Advantage grows with text length, consistent with design goal.
