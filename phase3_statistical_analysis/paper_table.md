### Table 1: Model Comparison (mean +/- std, 5 seeds)

| Model | Accuracy | Precision | Recall | Macro-F1 | Abs. Imp. (pp) | Rel. Imp. (%) |
|---|---|---|---|---|---|---|
| A: Transformer | 0.7870 +/- 0.0210 | 0.7873 +/- 0.0204 | 0.7878 +/- 0.0217 | 0.7860 +/- 0.0207 | -- | -- |
| B: +Position | 0.7967 +/- 0.0234 | 0.7965 +/- 0.0236 | 0.7971 +/- 0.0239 | 0.7959 +/- 0.0237 | +0.97 pp | +1.23% |
| C: +Simple Struct. | 0.8000 +/- 0.0233 | 0.7996 +/- 0.0228 | 0.8008 +/- 0.0237 | 0.7993 +/- 0.0235 | +1.30 pp | +1.65% |
| D: +SpiderWeb | 0.8080 +/- 0.0265 | 0.8084 +/- 0.0271 | 0.8075 +/- 0.0266 | 0.8070 +/- 0.0272 | +2.10 pp | +2.67% |
| E: +Random Bias | 0.7813 +/- 0.0218 | 0.7809 +/- 0.0216 | 0.7812 +/- 0.0218 | 0.7797 +/- 0.0218 | -0.57 pp | -0.72% |

### Table 2: Statistical Tests (paired, 5 seeds)

| Comparison | Metric | Mean Diff (pp) | 95% CI (pp) | t | p | Cohen's d | Wilcoxon p |
|---|---|---|---|---|---|---|---|
| SWeb vs Transformer | accuracy | +2.10 | [+1.00,+3.20] | 5.2963 | 0.0061 | 2.369 | 0.0625 |
| SWeb vs Transformer | macro_f1 | +2.11 | [+0.93,+3.28] | 4.9891 | 0.0075 | 2.231 | 0.0625 |
| SWeb vs RandomBias | accuracy | +2.67 | [+0.22,+5.11] | 3.0291 | 0.0388 | 1.355 | 0.1250 |
| SWeb vs RandomBias | macro_f1 | +2.73 | [+0.25,+5.21] | 3.0561 | 0.0378 | 1.367 | 0.0625 |
| SWeb vs SimpleStruct | accuracy | +0.80 | [+0.31,+1.29] | 4.4961 | 0.0109 | 2.011 | 0.0625 |
| SWeb vs SimpleStruct | macro_f1 | +0.77 | [+0.22,+1.32] | 3.8806 | 0.0178 | 1.735 | 0.0625 |