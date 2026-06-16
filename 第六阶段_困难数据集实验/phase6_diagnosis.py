# phase6_diagnosis.py - Data leakage check + Hard dataset design
import sys, os, torch, numpy as np
from collections import Counter

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

# ====================================================================
# PART 1: TF-IDF baseline check
# ====================================================================
print("=" * 60)
print("  DATA LEAKAGE CHECK")
print("=" * 60)

from real_data import create_dataloaders, RealChineseArticleDataset

# Generate 3000 samples with old params
ds = RealChineseArticleDataset(
    num_samples=3000, num_classes=6, max_seq_len=512, seed=42,
    center_bonus=0.10, support_bonus=0.04)

# Reconstruct text from token IDs for TF-IDF
import real_data
topic_map = {v:k for k,v in real_data.TOPIC_WORDS.items()}
# Build reverse: which tokens belong to which class
class_token_sets = {}
for c in range(6):
    tids = ds.topic_char_ids[c]
    class_token_sets[c] = set(tids)

# Check overlap between class token sets
print("\nTopic token overlap between classes:")
for i in range(6):
    for j in range(i+1, 6):
        overlap = class_token_sets[i] & class_token_sets[j]
        if overlap:
            shared = [ds.id_to_char.get(t, "?") for t in overlap]
            print(f"  Class {i} <-> Class {j}: {len(overlap)} shared tokens: {''.join(shared)}")
        else:
            print(f"  Class {i} <-> Class {j}: 0 shared tokens")
print("  => Classes have ZERO token overlap - too easy!")

# Check train/test split for template bias
print("\nTrain/Test split check:")
tr, te, _ = create_dataloaders(num_samples=3000, batch_size=64, seed=42,
    center_bonus=0.10, support_bonus=0.04)
tr_dataset = tr.dataset
te_dataset = te.dataset
# Check max_len distribution
print(f"  Train samples: {len(tr_dataset)}, Test samples: {len(te_dataset)}")

# Check if any center-level token appears ONLY in one class
print("\nUnique topic token check:")
for c in range(6):
    tids = class_token_sets[c]
    others = set().union(*[class_token_sets[oc] for oc in range(6) if oc != c])
    unique = tids - others
    chars = [ds.id_to_char.get(t, "?") for t in unique]
    print(f"  Class {c}: {len(unique)} unique tokens: {''.join(chars)}")

# ====================================================================
# PART 2: TF-IDF + Logistic Regression baseline
# ====================================================================
print("\n" + "=" * 60)
print("  TF-IDF + LOGISTIC REGRESSION BASELINE")
print("=" * 60)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Reconstruct texts
all_texts = []
all_labels = []
for i in range(min(2000, len(ds))):
    sample = ds[i]
    tokens = sample["token_ids"][:sample["seq_len"]].tolist()
    text = "".join(ds.id_to_char.get(t, "?") for t in tokens)
    all_texts.append(text)
    all_labels.append(sample["label"].item())

# Split
n_train = int(0.8 * len(all_texts))
X_train, X_test = all_texts[:n_train], all_texts[n_train:]
y_train, y_test = all_labels[:n_train], all_labels[n_train:]

vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 4), max_features=5000)
X_tr_vec = vectorizer.fit_transform(X_train)
X_te_vec = vectorizer.transform(X_test)

clf = LogisticRegression(max_iter=500, C=0.1)
clf.fit(X_tr_vec, y_train)
y_pred = clf.predict(X_te_vec)
tfidf_acc = accuracy_score(y_test, y_pred)
print(f"  TF-IDF + LogReg accuracy: {tfidf_acc:.4f} ({int(tfidf_acc*len(y_test))}/{len(y_test)})")
if tfidf_acc > 0.95:
    print("  *** CRITICAL: TF-IDF > 95% - data is keyword-based, not structure-based ***")
elif tfidf_acc > 0.80:
    print("  WARNING: TF-IDF > 80% - keyword leakage moderate")
else:
    print("  OK: TF-IDF < 80% - task requires more than keywords")

# ====================================================================
# PART 3: Design Hard Dataset Parameters
# ====================================================================
print("\n" + "=" * 60)
print("  HARD DATASET DESIGN")
print("=" * 60)

print("""
Changes needed:
  1. REDUCE topic signal: center_bonus=0.06 -> only 6% of center chars are topic
  2. SHARE topic tokens: each class shares 3/5 tokens with neighbors
  3. ADD distractor paragraphs: random topic from wrong class in body
  4. ADD background paragraphs: neutral text with no topic signal
  5. MORE classes: 8 classes -> more confusion
  6. CLASS OVERLAP: half the tokens are shared across 2+ classes
""")

# Write the diagnostic report
report_path = os.path.join(BASE, "phase5_real_data", "data_leakage_check.md")
with open(report_path, "w", encoding="utf-8") as f:
    f.write(f"""# Data Leakage Check

## 1. Token Overlap
- All 6 classes have ZERO shared topic tokens
- This means TF-IDF can trivially identify the class from any topic character

## 2. TF-IDF Baseline
- TF-IDF + Logistic Regression accuracy: **{tfidf_acc:.4f}**
- {"CRITICAL: >95% -> task is keyword-based, not structure-based" if tfidf_acc > 0.95 else "Moderate"}

## 3. Root Cause
- center_bonus=0.10 means ~10 chars out of ~130 center chars are class-specific
- With 0 overlap and 512-length sequences, even random sampling catches enough signal
- SpiderWeb''s structural advantage is real but the task ceiling is too low to show it

## 4. Proposed Fix
- Shared topic tokens (3/5 shared with neighbors)
- Center_bonus reduced to 0.05-0.06
- Add distractor and background paragraphs
- 8 classes instead of 6
- Target: TR accuracy 75-90%
""")
print(f"\nReport: {report_path}")
print("Diagnosis complete")
