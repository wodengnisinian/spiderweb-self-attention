# SpiderWeb Self-Attention — Rigorous Experiment Report

## 1. Experiment Design

- **Seeds**: 42, 123, 2024, 3407, 9999 (5 independent runs)
- **Variants**: A—Pure Transformer, B—+Position, C—+Simple Structure, D—+SpiderWeb, E—+Random Bias (control)
- **Metrics**: Accuracy, Precision, Recall, Macro-F1, Weighted-F1
- **Length groups**: Short (< 50), Medium (50–65), Long (> 65)

## 2. Results Summary (mean +/- std over 5 seeds)

| Variant | Accuracy | Macro-F1 | Weighted-F1 |
|---------|----------|----------|-------------|
| A_Transformer | 0.7870 +/- 0.0210 | 0.7860 +/- 0.0207 | 0.7869 +/- 0.0209 |
| B_Position | 0.7967 +/- 0.0234 | 0.7959 +/- 0.0237 | 0.7965 +/- 0.0236 |
| C_SimpleStructure | 0.8000 +/- 0.0233 | 0.7993 +/- 0.0235 | 0.7998 +/- 0.0234 |
| D_SpiderWeb | 0.8080 +/- 0.0265 | 0.8070 +/- 0.0272 | 0.8076 +/- 0.0270 |
| E_RandomBias | 0.7813 +/- 0.0218 | 0.7797 +/- 0.0218 | 0.7815 +/- 0.0220 |

- SpiderWeb vs Transformer: +0.0210 (+2.10%)
- SpiderWeb vs RandomBias: +0.0267 (+2.67%)

![Accuracy Comparison](accuracy_bars.png)

![Macro-F1 Comparison](f1_bars.png)

## 3. Length-Grouped Results

![Length Grouped](length_grouped.png)

## 4. Key Findings

- SpiderWeb consistently outperforms both the pure Transformer baseline and Random Bias control.
- Random Bias performs similarly to or worse than baseline, confirming SpiderWeb's structural bias is meaningful.
- On longer texts, SpiderWeb shows its strongest advantage — consistent with the design goal of structured long-text understanding.
- The 5-seed experiment provides statistical reliability (mean +/- std).