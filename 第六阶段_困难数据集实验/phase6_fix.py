# phase6_fix.py - Phase 6 Correction: Training stability fix
# Changes: 5 epochs, lr=5e-4, class_weight in loss, dev set + best ckpt, pred distribution
import csv, os, sys, time, torch, numpy as np
from collections import defaultdict, Counter

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from hard_data import create_dataloaders, build_m_web, HardChineseDataset
from model import create_model

OUT = os.path.join(BASE, "phase6_hard")
os.makedirs(OUT, exist_ok=True)

SEEDS = [42, 123, 2024]
N_CLASSES = 8
MAX_LEN = 256
BS = 8
EPOCHS = 5
SAMPLES = 2000
CENTER_BONUS = 0.05
SUPPORT_BONUS = 0.03
LR = 5e-4

print("=" * 60)
print("  PHASE 6 FIX: Stable Training")
print(f"  epochs={EPOCHS}, lr={LR}, dev+best_ckpt, class_weight")
print("=" * 60)

# TF-IDF check
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
ds_check = HardChineseDataset(num_samples=1000, num_classes=N_CLASSES, max_seq_len=MAX_LEN, seed=42,
    center_bonus=CENTER_BONUS, support_bonus=SUPPORT_BONUS, distractor_prob=0.15, background_prob=0.10)
texts = []; labels = []
for i in range(800):
    s = ds_check[i]; tokens = s["token_ids"][:s["seq_len"]].tolist()
    texts.append("".join(ds_check.id_to_char.get(t,"?") for t in tokens)); labels.append(s["label"].item())
n_tr = int(0.8 * len(texts))
vec = TfidfVectorizer(analyzer='char', ngram_range=(2,4), max_features=5000)
clf = LogisticRegression(max_iter=500, C=0.1)
clf.fit(vec.fit_transform(texts[:n_tr]), labels[:n_tr])
tfidf_acc = accuracy_score(labels[n_tr:], clf.predict(vec.transform(texts[n_tr:])))
print(f"TF-IDF: {tfidf_acc:.4f} {'FAIL (>95%)' if tfidf_acc>0.95 else 'PASS'}")

# Results CSV
CSV_PATH = os.path.join(OUT, "phase6_hard_results.csv")
DIST_PATH = os.path.join(OUT, "prediction_distribution.csv")
FIELDS = ["seed","model","accuracy","macro_f1","n_correct","n_total","train_time_s","best_epoch"]
DIST_FIELDS = ["seed","model","collapse"] + [f"pred_class_{c}" for c in range(N_CLASSES)]

with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
    csv.DictWriter(f, fieldnames=FIELDS).writeheader()
with open(DIST_PATH, "w", newline="", encoding="utf-8") as f:
    csv.DictWriter(f, fieldnames=DIST_FIELDS).writeheader()

for si, seed in enumerate(SEEDS):
    print(f"\n--- Seed {seed} ({si+1}/{len(SEEDS)}) ---")
    t_start = time.time()

    torch.manual_seed(seed)
    tr_ld, te_ld, ds = create_dataloaders(
        num_samples=SAMPLES, batch_size=BS, num_classes=N_CLASSES, max_seq_len=MAX_LEN, seed=seed,
        center_bonus=CENTER_BONUS, support_bonus=SUPPORT_BONUS, distractor_prob=0.15, background_prob=0.10)

    # Split test into dev (200) + test (200)
    te_indices = list(range(len(te_ld.dataset)))
    half = len(te_indices) // 2
    dev_subset = torch.utils.data.Subset(te_ld.dataset, te_indices[:half])
    test_subset = torch.utils.data.Subset(te_ld.dataset, te_indices[half:])
    dev_loader = torch.utils.data.DataLoader(dev_subset, batch_size=BS, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_subset, batch_size=BS, shuffle=False)

    # Compute class weights from training data
    label_counts = Counter()
    for batch in tr_ld:
        for l in batch["label"].numpy():
            label_counts[int(l)] += 1
    total = sum(label_counts.values())
    class_weights = torch.tensor([total / max(label_counts.get(c, 1), 1) for c in range(N_CLASSES)], dtype=torch.float)

    for model_type, bias_mode, build_fn, lam in [
        ("A_Transformer", "none", None, 0.0),
        ("E_SpiderWeb", "full", build_m_web, 0.5),
    ]:
        torch.manual_seed(seed * 7 + 13)
        m = create_model(bias_mode=bias_mode, vocab_size=ds.vocab_size, d_model=128,
            n_heads=4, n_layers=2, d_ff=512, n_classes=N_CLASSES, max_len=MAX_LEN).to("cpu")
        opt = torch.optim.Adam(m.parameters(), lr=LR, weight_decay=1e-5)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
        crit = torch.nn.CrossEntropyLoss(weight=class_weights)

        best_dev_acc = 0; best_state = None; best_epoch = 0
        train_losses = []; dev_accs = []

        for ep in range(EPOCHS):
            m.train(); ep_loss = 0.0; ep_total = 0
            for batch in tr_ld:
                tids = batch["token_ids"]; labs = batch["label"]
                mask = (tids != 0).unsqueeze(1).unsqueeze(2)
                M = build_fn(batch["levels"], batch["segments"], batch["seq_len"]) if build_fn else None
                loss = crit(m(tids, mask, M, lam), labs)
                opt.zero_grad(); loss.backward(); opt.step()
                ep_loss += loss.item() * tids.size(0); ep_total += tids.size(0)
            sched.step()
            train_losses.append(ep_loss / ep_total)

            # Dev eval
            m.eval(); dev_preds, dev_labels = [], []
            with torch.no_grad():
                for batch in dev_loader:
                    tids = batch["token_ids"]; labs = batch["label"]
                    mask = (tids != 0).unsqueeze(1).unsqueeze(2)
                    M = build_fn(batch["levels"], batch["segments"], batch["seq_len"]) if build_fn else None
                    logits = m(tids, mask, M, lam)
                    dev_preds.append(logits.argmax(dim=1).numpy()); dev_labels.append(labs.numpy())
            dp = np.concatenate(dev_preds); dl = np.concatenate(dev_labels)
            dev_acc = np.mean(dp == dl); dev_accs.append(dev_acc)
            if dev_acc > best_dev_acc:
                best_dev_acc = dev_acc; best_epoch = ep + 1
                best_state = {k: v.cpu().clone() for k, v in m.state_dict().items()}
            print(f"    {model_type} ep{ep+1}: loss={train_losses[-1]:.4f} dev={dev_acc:.4f}" + (" *" if dev_acc == best_dev_acc else ""))

        # Restore best ckpt and evaluate on test
        m.load_state_dict(best_state)
        m.eval(); test_preds, test_labels = [], []
        with torch.no_grad():
            for batch in test_loader:
                tids = batch["token_ids"]; labs = batch["label"]
                mask = (tids != 0).unsqueeze(1).unsqueeze(2)
                M = build_fn(batch["levels"], batch["segments"], batch["seq_len"]) if build_fn else None
                logits = m(tids, mask, M, lam)
                test_preds.append(logits.argmax(dim=1).numpy()); test_labels.append(labs.numpy())

        p = np.concatenate(test_preds); l = np.concatenate(test_labels)
        acc = np.mean(p == l); n_total = len(p)

        # Macro F1
        f1s = []
        for c in range(N_CLASSES):
            tp = np.sum((p == c) & (l == c)); fp = np.sum((p == c) & (l != c)); fn = np.sum((p != c) & (l == c))
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0; rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0)
        macro_f1 = np.mean(f1s)

        # Prediction distribution
        pred_dist = Counter(int(x) for x in p)
        total_preds = sum(pred_dist.values())
        dist_row = {"seed": seed, "model": model_type,
                    "collapse": "YES" if max(pred_dist.values(), default=0) > 0.5 * total_preds else "NO"}
        for c in range(N_CLASSES):
            dist_row[f"pred_class_{c}"] = pred_dist.get(c, 0)
        with open(DIST_PATH, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=DIST_FIELDS).writerow(dist_row)

        elapsed = time.time() - t_start
        row = {"seed": seed, "model": model_type, "accuracy": acc, "macro_f1": macro_f1,
               "n_correct": int(acc * n_total), "n_total": n_total,
               "train_time_s": elapsed, "best_epoch": best_epoch}
        with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writerow(row)

        collapse_status = "COLLAPSE!" if dist_row["collapse"] == "YES" else "OK"
        top_pred = max(pred_dist, key=pred_dist.get) if pred_dist else -1
        print(f"  -> TEST acc={acc:.4f} f1={macro_f1:.4f} best_ep={best_epoch} [{collapse_status}] pred_{top_pred}={pred_dist.get(top_pred,0)}")

# Summary
print(f"\n{'='*60}")
print(f"  PHASE 6 FIX SUMMARY")
print(f"{'='*60}")
with open(CSV_PATH, "r", encoding="utf-8") as f:
    all_rows = list(csv.DictReader(f))
groups = defaultdict(list)
for r in all_rows: groups[r["model"]].append((float(r["accuracy"]), float(r["macro_f1"])))
for name in ["A_Transformer", "E_SpiderWeb"]:
    accs = [x[0] for x in groups[name]]; f1s = [x[1] for x in groups[name]]
    print(f"  {name:<20s} acc={np.mean(accs):.4f}+/-{np.std(accs):.4f} f1={np.mean(f1s):.4f}+/-{np.std(f1s):.4f}")
tr_m = np.mean([x[0] for x in groups["A_Transformer"]])
sw_m = np.mean([x[0] for x in groups["E_SpiderWeb"]])
print(f"  Delta: {(sw_m-tr_m)*100:+.2f}pp")
# Show collapse status
print(f"\n  Collapse check:")
for r in csv.DictReader(open(DIST_PATH, encoding="utf-8")):
    print(f"    seed={r['seed']} {r['model']:<20s} collapse={r['collapse']}")
print(f"\n{'='*60}")
print(f"  DONE")
print(f"{'='*60}")
