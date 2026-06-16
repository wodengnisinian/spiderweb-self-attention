# Data Leakage Check

## TF-IDF Baseline
- TF-IDF + Logistic Regression accuracy: **0.1100**
- Classification: PASS - <95% - requires more than keyword matching

## Hard Dataset Design
- center_bonus=0.05 (was 0.10 in Phase 5)
- support_bonus=0.03
- distractor_prob=0.15
- background_prob=0.1
- 8 classes with shared topic tokens
- All classes share common background characters
- Distractor: wrong-class topic tokens in center/support paragraphs
- Background: paragraphs with zero topic signal
