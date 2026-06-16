# phase5_experiment.py - Phase 5 V3: Incremental, 3 seeds, 6 epochs, 1000 samples
import csv, os, sys, time, torch, numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from real_data import create_dataloaders, build_m_web
from model import create_model

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams.update({"figure.figsize":(12,5),"font.size":11,"savefig.dpi":150,"savefig.bbox":"tight"})

OUT = os.path.join(BASE, "phase5_real_data")
os.makedirs(OUT, exist_ok=True)

SEEDS = [42, 123, 2024]
N_CLASSES = 6
MAX_LEN = 512
SAMPLES = 1000
BS = 4
EPOCHS = 6
CENTER_BONUS = 0.10
SUPPORT_BONUS = 0.04

import real_data
real_data.TOPIC_WORDS = {
    0: ["数据","算法","网络","系统","编程"],
    1: ["经济","市场","投资","金融","增长"],
    2: ["教育","学生","学校","学习","课程"],
    3: ["体育","比赛","冠军","球队","决赛"],
    4: ["电影","音乐","明星","演出","节目"],
    5: ["医疗","健康","医院","疾病","治疗"],
}

print("=" * 60)
print("  PHASE 5 V3: Incremental save, 3 seeds, 1000 samples, 6 epochs")
print(f"  center_bonus={CENTER_BONUS}, support_bonus={SUPPORT_BONUS}")
print("=" * 60)

CSV_PATH = os.path.join(OUT, "phase5_results_v3.csv")
FIELDS = ["seed","model","accuracy","n_correct","n_total","train_time_s","acc_short","acc_medium","acc_long"]

# Write header
with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
    csv.DictWriter(f, fieldnames=FIELDS).writeheader()

all_tr_cm = np.zeros((N_CLASSES, N_CLASSES), dtype=int)
all_sw_cm = np.zeros((N_CLASSES, N_CLASSES), dtype=int)

for si, seed in enumerate(SEEDS):
    print(f"\n--- Seed {seed} ({si+1}/{len(SEEDS)}) ---")
    t0 = time.time()

    torch.manual_seed(seed)
    tr_ld, te_ld, ds = create_dataloaders(
        num_samples=SAMPLES, batch_size=BS, num_classes=N_CLASSES,
        max_seq_len=MAX_LEN, seed=seed,
        center_bonus=CENTER_BONUS, support_bonus=SUPPORT_BONUS)

    for model_type, bias_mode, build_fn, lam in [
        ("Transformer", "none", None, 0.0),
        ("SpiderWeb", "full", build_m_web, 0.5),
    ]:
        torch.manual_seed(seed * 7 + 13)
        m = create_model(
            bias_mode=bias_mode, vocab_size=ds.vocab_size, d_model=128,
            n_heads=4, n_layers=2, d_ff=512, n_classes=N_CLASSES, max_len=MAX_LEN
        ).to("cpu")

        opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-5)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
        crit = torch.nn.CrossEntropyLoss()

        for ep in range(EPOCHS):
            m.train()
            ep_loss = 0.0; ep_total = 0
            for batch in tr_ld:
                tids = batch["token_ids"]; labs = batch["label"]
                mask = (tids != 0).unsqueeze(1).unsqueeze(2)
                M = build_fn(batch["levels"], batch["segments"], batch["seq_len"]) if build_fn else None
                loss = crit(m(tids, mask, M, lam), labs)
                opt.zero_grad(); loss.backward(); opt.step()
                ep_loss += loss.item() * tids.size(0); ep_total += tids.size(0)
            sched.step()
            if (ep + 1) % 2 == 0:
                print(f"    epoch {ep+1:2d}/{EPOCHS}  loss={ep_loss/ep_total:.4f}")

        m.eval(); preds, labels, seq_lens = [], [], []
        with torch.no_grad():
            for batch in te_ld:
                tids = batch["token_ids"]; labs = batch["label"]
                mask = (tids != 0).unsqueeze(1).unsqueeze(2)
                M = build_fn(batch["levels"], batch["segments"], batch["seq_len"]) if build_fn else None
                logits = m(tids, mask, M, lam)
                preds.append(logits.argmax(dim=1).numpy()); labels.append(labs.numpy())
                seq_lens.append(batch["seq_len"].clone().detach().numpy())

        p = np.concatenate(preds); l = np.concatenate(labels); sl = np.concatenate(seq_lens)
        acc = np.mean(p == l); n_total = len(p)

        cm = np.zeros((N_CLASSES, N_CLASSES), dtype=int)
        for t, pp in zip(l, p): cm[t, pp] += 1
        if model_type == "Transformer": all_tr_cm += cm
        else: all_sw_cm += cm

        lg = {}
        for gn, lo, hi in [("short", 0, 300), ("medium", 300, 450), ("long", 450, 9999)]:
            msk = (sl >= lo) & (sl < hi)
            lg[gn] = (np.mean(p[msk] == l[msk]), msk.sum()) if msk.sum() > 0 else (None, 0)

        elapsed = time.time() - t0
        row = {"seed": seed, "model": model_type, "accuracy": acc,
               "n_correct": int(np.trace(cm)), "n_total": n_total, "train_time_s": elapsed}
        for gn in lg: row[f"acc_{gn}"] = lg[gn][0] if lg[gn][0] is not None else 0

        # INCREMENTAL SAVE
        with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writerow(row)

        sh = f"short={lg['short'][0]:.4f}" if lg["short"][0] else ""
        md = f" med={lg['medium'][0]:.4f}" if lg["medium"][0] else ""
        lo = f" long={lg['long'][0]:.4f}" if lg["long"][0] else ""
        print(f"  {model_type:<15s} acc={acc:.4f} ({int(np.trace(cm))}/{n_total}) {sh}{md}{lo}  [{elapsed:.0f}s]  SAVED")
        t0 = time.time()

# ================================================================
# Summary (from saved CSV)
# ================================================================
with open(CSV_PATH, "r", encoding="utf-8") as f:
    all_rows = list(csv.DictReader(f))

for r in all_rows:
    for k in ["accuracy","n_correct","n_total","train_time_s","acc_short","acc_medium","acc_long"]:
        r[k] = float(r[k])
    r["seed"] = int(float(r["seed"]))

tr_rows = [r for r in all_rows if r["model"] == "Transformer"]
sw_rows = [r for r in all_rows if r["model"] == "SpiderWeb"]
tr_total = sum(r["n_correct"] for r in tr_rows)
sw_total = sum(r["n_correct"] for r in sw_rows)
total = sum(r["n_total"] for r in tr_rows)
TR, SW = tr_total / total, sw_total / total
D = SW - TR
tr_accs = [r["accuracy"] for r in tr_rows]
sw_accs = [r["accuracy"] for r in sw_rows]

print(f"\n{'='*60}")
print(f"  PHASE 5 V3 SUMMARY")
print(f"{'='*60}")
for i, s in enumerate(SEEDS):
    d = (sw_accs[i] - tr_accs[i]) * 100
    print(f"  seed {s:4d}: TR={tr_accs[i]:.4f}  SW={sw_accs[i]:.4f}  delta={sw_accs[i]-tr_accs[i]:+.4f} ({d:+.2f}pp)")
print(f"\n  MEAN:  TR={np.mean(tr_accs):.4f}+/-{np.std(tr_accs):.4f}  SW={np.mean(sw_accs):.4f}+/-{np.std(sw_accs):.4f}")
print(f"  AGG:   TR={TR:.4f} ({tr_total}/{total})  SW={SW:.4f} ({sw_total}/{total})")
print(f"  DELTA: {D:+.4f} ({D*100:.2f} pp, {D/TR*100:.2f}% rel)")

# Length-grouped
print(f"\n  Length-grouped:")
for gn in ["short", "medium", "long"]:
    tr_lg = np.mean([r[f"acc_{gn}"] for r in tr_rows if r[f"acc_{gn}"] > 0])
    sw_lg = np.mean([r[f"acc_{gn}"] for r in sw_rows if r[f"acc_{gn}"] > 0])
    print(f"    {gn:8s}: TR={tr_lg:.4f}  SW={sw_lg:.4f}  delta={sw_lg-tr_lg:+.4f}")

# CMs
def save_cm(cm, title, fn):
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt=".0f", cmap="Blues", ax=ax,
                xticklabels=range(N_CLASSES), yticklabels=range(N_CLASSES))
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title, fontsize=11)
    plt.tight_layout(); p = os.path.join(OUT, fn); fig.savefig(p); plt.close(fig); return p

save_cm(all_tr_cm, f"A. Transformer ({int(np.trace(all_tr_cm))}/{total}, acc={TR:.4f})", "confusion_matrix_transformer_phase5_v3.png")
save_cm(all_sw_cm, f"D. SpiderWeb ({int(np.trace(all_sw_cm))}/{total}, acc={SW:.4f})", "confusion_matrix_spiderweb_phase5_v3.png")

# Report
lines = []
lines.append("# Phase 5 V3: Real Chinese Long Article Experiment")
lines.append("")
lines.append(f"center_bonus={CENTER_BONUS}, support_bonus={SUPPORT_BONUS}, max_len={MAX_LEN}")
lines.append(f"Seeds: {SEEDS}, Samples: {SAMPLES}, Epochs: {EPOCHS}")
lines.append("")
lines.append("## Per-Seed")
lines.append("")
lines.append("| Seed | Transformer | SpiderWeb | Delta |")
lines.append("|------|:-----------:|:---------:|:-----:|")
for i, s in enumerate(SEEDS):
    d = (sw_accs[i] - tr_accs[i]) * 100
    lines.append(f"| {s} | {tr_accs[i]:.4f} | {sw_accs[i]:.4f} | {sw_accs[i]-tr_accs[i]:+.4f} ({d:+.2f}pp) |")
lines.append(f"| **Mean** | **{np.mean(tr_accs):.4f}+/-{np.std(tr_accs):.4f}** | **{np.mean(sw_accs):.4f}+/-{np.std(sw_accs):.4f}** | **{np.mean(sw_accs)-np.mean(tr_accs):+.4f}** |")
lines.append("")
lines.append("| Model | Correct/Total | Accuracy | Abs. Imp. | Rel. Imp. |")
lines.append("|---|---|---|---|---|")
lines.append(f"| Transformer | {tr_total}/{total} | {TR:.4f} | baseline | -- |")
lines.append(f"| **SpiderWeb** | **{sw_total}/{total}** | **{SW:.4f}** | **{D*100:+.2f} pp** | **{D/TR*100:+.2f}%** |")
lines.append("")
lines.append("## Comparison: Phase 4 vs Phase 5")
lines.append("")
lines.append("| Metric | Phase 4 (80-char synth) | Phase 5 (512-char real) |")
lines.append("|--------|------------------------|------------------------|")
lines.append(f"| TR | 0.7853 | {TR:.4f} |")
lines.append(f"| SW | 0.8150 | {SW:.4f} |")
lines.append(f"| Delta | +2.97pp | {D*100:+.2f}pp |")
lines.append("")
lines.append("![TR CM](confusion_matrix_transformer_phase5_v3.png)")
lines.append("![SW CM](confusion_matrix_spiderweb_phase5_v3.png)")

with open(os.path.join(OUT, "phase5_report_v3.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print(f"\nReport: {os.path.join(OUT, 'phase5_report_v3.md')}")
print(f"\n{'='*60}")
print(f"  PHASE 5 V3 DONE - Delta = {D:+.4f} ({D*100:.2f}pp)")
print(f"{'='*60}")
