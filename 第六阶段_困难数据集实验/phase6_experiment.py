# phase6_experiment.py - Phase 6: Fast version
# Max_len=256, batch=8, 2 seeds, 3 epochs, 5 models
import csv, os, sys, time, torch, numpy as np
from collections import defaultdict

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from hard_data import create_dataloaders, build_m_web, HardChineseDataset
from model import create_model

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams.update({"figure.figsize":(12,5),"font.size":11,"savefig.dpi":150,"savefig.bbox":"tight"})

OUT = os.path.join(BASE, "phase6_hard")
os.makedirs(OUT, exist_ok=True)

SEEDS = [42, 123, 2024]
N_CLASSES = 8
MAX_LEN = 256
BS = 8
EPOCHS = 3
SAMPLES = 2000
CENTER_BONUS = 0.05
SUPPORT_BONUS = 0.03

print("=" * 60)
print("  PHASE 6: Fast Version")
print(f"  Max_len={MAX_LEN}, batch={BS}, seeds={len(SEEDS)}, epochs={EPOCHS}")
print("=" * 60)

# ================ TF-IDF CHECK ================
print("\n--- TF-IDF Baseline ---")
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
print(f"  TF-IDF accuracy: {tfidf_acc:.4f} {'*** TOO EASY' if tfidf_acc>0.95 else 'OK'}")

# ================ EXPERIMENT ================
CSV_PATH = os.path.join(OUT, "phase6_hard_results.csv")
FIELDS = ["seed","model","accuracy","macro_f1","n_correct","n_total","train_time_s"]

with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
    csv.DictWriter(f, fieldnames=FIELDS).writeheader()

VARIANTS = [
    ("A_Transformer", "none", None, 0.0),
    ("B_Position", "pos", "pos", 0.5),
    ("C_RandomBias", "full", "random", 0.5),
    ("D_SimpleStructure", "simple", "simple", 0.5),
    ("E_SpiderWeb", "full", "spiderweb", 0.5),
]

for si, seed in enumerate(SEEDS):
    print(f"\n--- Seed {seed} ({si+1}/{len(SEEDS)}) ---")
    t_start = time.time()

    torch.manual_seed(seed)
    tr_ld, te_ld, ds = create_dataloaders(
        num_samples=SAMPLES, batch_size=BS, num_classes=N_CLASSES, max_seq_len=MAX_LEN, seed=seed,
        center_bonus=CENTER_BONUS, support_bonus=SUPPORT_BONUS,
        distractor_prob=0.15, background_prob=0.10)

    for var_name, bias_mode, build_key, lam in VARIANTS:
        # Build bias function
        if build_key == "spiderweb":
            build_fn = build_m_web
        elif build_key == "random":
            def make_random():
                def fn(levels, segs, sls):
                    B, N = levels.shape; valid = (levels >= 0).float()
                    vp = torch.bmm(valid.unsqueeze(-1), valid.unsqueeze(1))
                    r = build_m_web(levels, segs, sls)
                    rand = torch.randn(B, N, N) * r.std().item() + r.mean().item()
                    return rand * vp
                return fn
            build_fn = make_random()
        elif build_key == "pos":
            def make_pos():
                def fn(levels, segs, sls):
                    B, N = levels.shape; valid = (levels >= 0).float()
                    vp = torch.bmm(valid.unsqueeze(-1), valid.unsqueeze(1))
                    ss = torch.tensor(segs).clone(); ss[ss < 0] = -999
                    same = (ss.unsqueeze(-1) == ss.unsqueeze(1)).float()
                    adj = ((ss.unsqueeze(-1) - ss.unsqueeze(1)).abs() == 1).float()
                    return (1.5*same + 0.75*adj) * vp
                return fn
            build_fn = make_pos()
        elif build_key == "simple":
            def make_simple():
                def fn(levels, segs, sls):
                    B, N = levels.shape; valid = (levels >= 0).float()
                    vp = torch.bmm(valid.unsqueeze(-1), valid.unsqueeze(1))
                    sl = levels.clone(); sl[sl < 0] = 0
                    ld = (sl.unsqueeze(-1) - sl.unsqueeze(1)).abs().float()
                    Mh = -1.0 * ld * vp
                    ss = torch.tensor(segs).clone(); ss[ss < 0] = -999
                    same = (ss.unsqueeze(-1) == ss.unsqueeze(1)).float()
                    adj = ((ss.unsqueeze(-1) - ss.unsqueeze(1)).abs() == 1).float()
                    return Mh + (1.5*same + 0.75*adj) * vp
                return fn
            build_fn = make_simple()
        else:
            build_fn = None

        torch.manual_seed(seed * 7 + 13)
        m = create_model(bias_mode=bias_mode, vocab_size=ds.vocab_size, d_model=128,
            n_heads=4, n_layers=2, d_ff=512, n_classes=N_CLASSES, max_len=MAX_LEN).to("cpu")
        opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-5)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
        crit = torch.nn.CrossEntropyLoss()

        for ep in range(EPOCHS):
            m.train()
            for batch in tr_ld:
                tids = batch["token_ids"]; labs = batch["label"]
                mask = (tids != 0).unsqueeze(1).unsqueeze(2)
                M = build_fn(batch["levels"], batch["segments"], batch["seq_len"]) if build_fn else None
                loss = crit(m(tids, mask, M, lam), labs)
                opt.zero_grad(); loss.backward(); opt.step()
            sched.step()

        m.eval(); preds, labels = [], []
        with torch.no_grad():
            for batch in te_ld:
                tids = batch["token_ids"]; labs = batch["label"]
                mask = (tids != 0).unsqueeze(1).unsqueeze(2)
                M = build_fn(batch["levels"], batch["segments"], batch["seq_len"]) if build_fn else None
                logits = m(tids, mask, M, lam)
                preds.append(logits.argmax(dim=1).numpy()); labels.append(labs.numpy())

        p = np.concatenate(preds); l = np.concatenate(labels)
        acc = np.mean(p == l); n_total = len(p)

        f1s = []
        for c in range(N_CLASSES):
            tp = np.sum((p == c) & (l == c))
            fp = np.sum((p == c) & (l != c))
            fn = np.sum((p != c) & (l == c))
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
            f1s.append(f1)
        macro_f1 = np.mean(f1s)

        elapsed = time.time() - t_start
        row = {"seed": seed, "model": var_name, "accuracy": acc, "macro_f1": macro_f1,
               "n_correct": int(acc * n_total), "n_total": n_total, "train_time_s": elapsed}

        with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writerow(row)

        print(f"  {var_name:<20s} acc={acc:.4f}  f1={macro_f1:.4f}  [{elapsed:.0f}s]")
        t_start = time.time()

# ================ SUMMARY ================
print(f"\n{'='*60}")
print(f"  PHASE 6 SUMMARY")
print(f"{'='*60}")

with open(CSV_PATH, "r", encoding="utf-8") as f:
    all_rows = list(csv.DictReader(f))

groups = defaultdict(list)
for r in all_rows:
    groups[r["model"]].append((float(r["accuracy"]), float(r["macro_f1"])))

ORDER = ["A_Transformer","B_Position","C_RandomBias","D_SimpleStructure","E_SpiderWeb"]
LABELS = ["A: Transformer","B: +Position","C: +RandomBias","D: +SimpleStruct","E: +SpiderWeb"]

for name, label in zip(ORDER, LABELS):
    accs = [x[0] for x in groups[name]]
    f1s = [x[1] for x in groups[name]]
    print(f"  {label:<20s} acc={np.mean(accs):.4f}+/-{np.std(accs):.4f}  f1={np.mean(f1s):.4f}")

tr_m = np.mean([x[0] for x in groups["A_Transformer"]])
sw_m = np.mean([x[0] for x in groups["E_SpiderWeb"]])
print(f"\n  Delta: {(sw_m-tr_m)*100:+.2f}pp")
print(f"  TF-IDF: {tfidf_acc:.4f}")
print(f"  Task difficulty (TR acc): {tr_m:.4f}")

# Build confusion matrices
all_tr_cm = np.zeros((N_CLASSES,N_CLASSES),dtype=int)
all_sw_cm = np.zeros((N_CLASSES,N_CLASSES),dtype=int)

# Re-run just TR and SW from seed 2024 for CM
for seed in [2024]:
    torch.manual_seed(seed)
    _, te_ld, ds = create_dataloaders(num_samples=SAMPLES,batch_size=BS,num_classes=N_CLASSES,
        max_seq_len=MAX_LEN,seed=seed,center_bonus=CENTER_BONUS,support_bonus=SUPPORT_BONUS,
        distractor_prob=0.15,background_prob=0.10)
    for model_type, bias_mode, build_fn, lam in [("TR","none",None,0.0),("SW","full",build_m_web,0.5)]:
        torch.manual_seed(seed*7+13)
        m=create_model(bias_mode=bias_mode,vocab_size=ds.vocab_size,d_model=128,n_heads=4,
            n_layers=2,d_ff=512,n_classes=N_CLASSES,max_len=MAX_LEN).to("cpu")
        opt=torch.optim.Adam(m.parameters(),lr=1e-3,weight_decay=1e-5)
        sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=3)
        crit=torch.nn.CrossEntropyLoss()
        for _ in range(3):
            m.train()
            for batch in tr_ld:
                tids=batch["token_ids"]; labs=batch["label"]
                mask=(tids!=0).unsqueeze(1).unsqueeze(2)
                M=build_fn(batch["levels"],batch["segments"],batch["seq_len"]) if build_fn else None
                loss=crit(m(tids,mask,M,lam),labs); opt.zero_grad(); loss.backward(); opt.step()
            sched.step()
        m.eval(); preds,labels=[],[]
        with torch.no_grad():
            for batch in te_ld:
                tids=batch["token_ids"]; labs=batch["label"]
                mask=(tids!=0).unsqueeze(1).unsqueeze(2)
                M=build_fn(batch["levels"],batch["segments"],batch["seq_len"]) if build_fn else None
                logits=m(tids,mask,M,lam)
                preds.append(logits.argmax(dim=1).numpy()); labels.append(labs.numpy())
        p=np.concatenate(preds); l=np.concatenate(labels)
        cm=np.zeros((N_CLASSES,N_CLASSES),dtype=int)
        for t,pp in zip(l,p): cm[t,pp]+=1
        if model_type=="TR": all_tr_cm+=cm
        else: all_sw_cm+=cm

def save_cm(cm,title,fn):
    fig,ax=plt.subplots(figsize=(8,7))
    sns.heatmap(cm,annot=True,fmt=".0f",cmap="Blues",ax=ax,xticklabels=range(N_CLASSES),yticklabels=range(N_CLASSES))
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title,fontsize=11)
    plt.tight_layout(); p=os.path.join(OUT,fn); fig.savefig(p); plt.close(fig); return p

save_cm(all_tr_cm,f"A. Transformer (seed 2024, {int(np.trace(all_tr_cm))}/{int(all_tr_cm.sum())})","confusion_matrix_transformer_phase6.png")
save_cm(all_sw_cm,f"E. SpiderWeb (seed 2024, {int(np.trace(all_sw_cm))}/{int(all_sw_cm.sum())})","confusion_matrix_spiderweb_phase6.png")

# Report
lines=[]
lines.append("# Phase 6: Hard Dataset Experiment Report")
lines.append("")
lines.append(f"**Setup**: max_len={MAX_LEN}, batch={BS}, seeds={SEEDS}, epochs={EPOCHS}")
lines.append(f"**Data**: center_bonus={CENTER_BONUS}, shared tokens, distractors, 8 classes")
lines.append("")
lines.append(f"**TF-IDF baseline**: {tfidf_acc:.4f} ({'KEYWORD-BASED FAIL' if tfidf_acc>0.95 else 'STRUCTURE-DEPENDENT PASS'})")
lines.append("")
lines.append("## Results")
lines.append("")
lines.append("| Model | Accuracy | Macro-F1 | Delta vs TR |")
lines.append("|---|---|---|---|")
for name, label in zip(ORDER, LABELS):
    accs=[x[0] for x in groups[name]]; f1s=[x[1] for x in groups[name]]
    d=np.mean(accs)-tr_m
    lines.append(f"| {label} | {np.mean(accs):.4f}+/-{np.std(accs):.4f} | {np.mean(f1s):.4f}+/-{np.std(f1s):.4f} | {d*100:+.2f}pp |")
lines.append("")
lines.append(f"**SpiderWeb vs Transformer: {(sw_m-tr_m)*100:+.2f}pp**")
lines.append("")
lines.append("![TR CM](confusion_matrix_transformer_phase6.png)")
lines.append("![SW CM](confusion_matrix_spiderweb_phase6.png)")

with open(os.path.join(OUT,"phase6_hard_report.md"),"w",encoding="utf-8") as f:
    f.write("\n".join(lines))

print(f"\nReport: {os.path.join(OUT,'phase6_hard_report.md')}")
print(f"CSV: {CSV_PATH}")
print(f"\n{'='*60}")
print(f"  PHASE 6 DONE - Delta={(sw_m-tr_m)*100:+.2f}pp")
print(f"{'='*60}")
