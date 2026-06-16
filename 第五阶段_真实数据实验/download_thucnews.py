# download_thucnews.py - Download and preprocess THUCNews for SpiderWeb
import os, sys, urllib.request, tarfile, glob, random, json, gzip, shutil
from collections import Counter

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, "data", "thucnews")
os.makedirs(DATA_DIR, exist_ok=True)

# ============================================================
# Step 1: Download THUCNews
# ============================================================
# THUCNews is hosted at thunlp.org. Try multiple mirrors.
URLS = [
    "http://thuctc.thunlp.org/THUCNews.zip",
]

ZIP_PATH = os.path.join(DATA_DIR, "THUCNews.zip")
EXTRACT_DIR = os.path.join(DATA_DIR, "THUCNews")

if os.path.exists(EXTRACT_DIR) and len(os.listdir(EXTRACT_DIR)) > 5:
    print("THUCNews already extracted at", EXTRACT_DIR)
else:
    print("Downloading THUCNews... this may take a while (~1GB)")
    try:
        urllib.request.urlretrieve(URLS[0], ZIP_PATH)
        print("Downloaded to", ZIP_PATH)
        import zipfile
        with zipfile.ZipFile(ZIP_PATH, 'r') as zf:
            zf.extractall(DATA_DIR)
        print("Extracted to", EXTRACT_DIR)
    except Exception as e:
        print("Download failed:", e)
        print("\nTHUCNews direct download may not work. Providing alternative approach...")
        print("Please manually download from: http://thuctc.thunlp.org/")
        print("Or use the built-in synthetic data as fallback.")
        sys.exit(1)

# ============================================================
# Step 2: Build character vocabulary
# ============================================================
# Chinese characters: collect from all articles
CHAR_COUNTER = Counter()
FILES_BY_CLASS = {}
class_dirs = [d for d in os.listdir(EXTRACT_DIR) if os.path.isdir(os.path.join(EXTRACT_DIR, d))]
print(f"\nFound {len(class_dirs)} classes: {class_dirs[:10]}")

for cls in class_dirs:
    cls_path = os.path.join(EXTRACT_DIR, cls)
    files = [f for f in os.listdir(cls_path) if f.endswith('.txt')]
    FILES_BY_CLASS[cls] = files
    print(f"  {cls}: {len(files)} files")

    # Sample 10% of files to build vocab
    for f in files[:max(10, len(files)//10)]:
        with open(os.path.join(cls_path, f), 'r', encoding='utf-8', errors='ignore') as fh:
            text = fh.read().strip()
            for ch in text:
                CHAR_COUNTER[ch] += 1

# Build vocabulary
all_chars = CHAR_COUNTER.most_common(8000)
# Keep all Chinese chars + common ASCII
char_to_id = {'[PAD]': 0, '[UNK]': 1}
id_to_char = {0: '[PAD]', 1: '[UNK]'}
idx = 2
for ch, cnt in all_chars:
    if idx >= 5000:
        break
    char_to_id[ch] = idx
    id_to_char[idx] = ch
    idx += 1

VOCAB_SIZE = len(char_to_id)
print(f"\nVocabulary size: {VOCAB_SIZE}")

# Save vocabulary
import json
with open(os.path.join(DATA_DIR, "vocab.json"), 'w', encoding='utf-8') as f:
    json.dump({"char_to_id": char_to_id, "id_to_char": {str(k): v for k, v in id_to_char.items()}}, f, ensure_ascii=False)

# ============================================================
# Step 3: Process articles into SpiderWeb format
# ============================================================
# Select top 6 classes (balanced, 800 samples each)
TOP_CLASSES = [c for c, fs in sorted(FILES_BY_CLASS.items(), key=lambda x: -len(x[1]))[:6]]
print(f"\nTop 6 classes: {TOP_CLASSES}")

CLASS_TO_LABEL = {c: i for i, c in enumerate(TOP_CLASSES)}
MAX_LEN = 512

# Process each article
all_samples = []
for cls in TOP_CLASSES:
    cls_path = os.path.join(EXTRACT_DIR, cls)
    files = [f for f in os.listdir(cls_path) if f.endswith('.txt')][:800]
    label = CLASS_TO_LABEL[cls]
    
    for f in files:
        with open(os.path.join(cls_path, f), 'r', encoding='utf-8', errors='ignore') as fh:
            text = fh.read().strip()
        
        # Convert to token IDs
        tokens = []
        for ch in text:
            if ch in char_to_id:
                tokens.append(char_to_id[ch])
            else:
                tokens.append(char_to_id['[UNK]'])
        
        n = len(tokens)
        if n < 50:
            continue
        
        # Construct levels using inverted pyramid rule
        # L0 (center): first 25% of tokens
        # L1 (support): next 35%
        # L2 (description): remaining 40%
        levels = []
        segments = []
        seg_id = 0
        
        l0_end = min(int(n * 0.25), MAX_LEN)
        l1_end = min(int(n * 0.60), MAX_LEN)
        
        for i in range(min(n, MAX_LEN)):
            if i < l0_end:
                levels.append(0)
            elif i < l1_end:
                levels.append(1)
            else:
                levels.append(2)
        
        # Truncate to MAX_LEN
        if len(tokens) > MAX_LEN:
            tokens = tokens[:MAX_LEN]
        else:
            tokens = tokens + [0] * (MAX_LEN - len(tokens))
        
        all_samples.append({
            "tokens": tokens,
            "levels": levels[:MAX_LEN] if len(levels) >= MAX_LEN else levels + [-1] * (MAX_LEN - len(levels)),
            "label": label,
            "seq_len": min(n, MAX_LEN),
            "class_name": cls,
        })

print(f"\nTotal processed samples: {len(all_samples)}")

# Save as JSON
with open(os.path.join(DATA_DIR, "processed_data.json"), 'w', encoding='utf-8') as f:
    json.dump(all_samples, f, ensure_ascii=False)

with open(os.path.join(DATA_DIR, "class_mapping.json"), 'w', encoding='utf-8') as f:
    json.dump(CLASS_TO_LABEL, f, ensure_ascii=False)

print("Preprocessing complete!")
print(f"  Vocabulary: {VOCAB_SIZE} chars")
print(f"  Classes: {len(TOP_CLASSES)}")
print(f"  Samples: {len(all_samples)}")
print(f"  Max seq length: {MAX_LEN}")
