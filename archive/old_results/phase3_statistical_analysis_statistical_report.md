# SpiderWeb Self-Attention — Statistical Analysis Report

## 1. Per-Seed Raw Results

See [per_seed_raw_results.csv](per_seed_raw_results.csv)

## 2. Model Comparison (mean +/- std)

| Model | Accuracy | Precision | Recall | Macro-F1 | Weighted-F1 | Abs. Imp. (pp) | Rel. Imp. (%) |
|---|---|---|---|---|---|---|---|
| A: Transformer | 0.7870 +/- 0.0210 | 0.7873 +/- 0.0204 | 0.7878 +/- 0.0217 | 0.7860 +/- 0.0207 | 0.7869 +/- 0.0209 | — | — |
| B: +Position | 0.7967 +/- 0.0234 | 0.7965 +/- 0.0236 | 0.7971 +/- 0.0239 | 0.7959 +/- 0.0237 | 0.7965 +/- 0.0236 | +0.97 pp | +1.23% |
| C: +Simple Struct. | 0.8000 +/- 0.0233 | 0.7996 +/- 0.0228 | 0.8008 +/- 0.0237 | 0.7993 +/- 0.0235 | 0.7998 +/- 0.0234 | +1.30 pp | +1.65% |
| D: +SpiderWeb | 0.8080 +/- 0.0265 | 0.8084 +/- 0.0271 | 0.8075 +/- 0.0266 | 0.8070 +/- 0.0272 | 0.8076 +/- 0.0270 | +2.10 pp | +2.67% |
| E: +Random Bias | 0.7813 +/- 0.0218 | 0.7809 +/- 0.0216 | 0.7812 +/- 0.0218 | 0.7797 +/- 0.0218 | 0.7815 +/- 0.0220 | -0.57 pp | -0.72% |

## 3. Statistical Tests

| Comparison | Metric | Mean Diff | 95% CI | t-stat | p-value (t) | Wilcoxon p |
|---|---|---|---|---|---|---|
| SpiderWeb vs Transformer | accuracy | +2.10 pp | [+1.00 pp, +3.20 pp] | 5.2963 | 0.0061 | 0.0625 |
| SpiderWeb vs Transformer | macro_f1 | +2.11 pp | [+0.93 pp, +3.28 pp] | 4.9891 | 0.0075 | 0.0625 |
| SpiderWeb vs Random Bias | accuracy | +2.67 pp | [+0.22 pp, +5.11 pp] | 3.0291 | 0.0388 | 0.1250 |
| SpiderWeb vs Random Bias | macro_f1 | +2.73 pp | [+0.25 pp, +5.21 pp] | 3.0561 | 0.0378 | 0.0625 |
| SpiderWeb vs Simple Struct. | accuracy | +0.80 pp | [+0.31 pp, +1.29 pp] | 4.4961 | 0.0109 | 0.0625 |
| SpiderWeb vs Simple Struct. | macro_f1 | +0.77 pp | [+0.22 pp, +1.32 pp] | 3.8806 | 0.0178 | 0.0625 |

## 4. Charts

![Accuracy Comparison](accuracy_bars.png)

![Macro-F1 Comparison](f1_bars.png)

![Length-Grouped Accuracy](length_grouped.png)

## 5. Key Findings

- SpiderWeb outperforms Transformer by **+2.10 percentage points** (relative improvement **+2.67%**).
- Random Bias underperforms baseline, confirming SpiderWeb's structure is meaningful.
- Statistical significance (paired t-test, n=5) shown in table above.
- Longer texts show larger SpiderWeb advantage, consistent with the design goal.