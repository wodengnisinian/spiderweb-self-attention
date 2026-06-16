# experiment.py - Rigorous Multi-Seed Ablation Study with Random Bias Control
#
# Runs 5 seeds x 5 variants, computes Accuracy / Precision / Recall / Macro-F1 / Weighted-F1,
# length-grouped metrics (short/medium/long), and saves CSV + JSON.

import csv
import json
import os
import time
import torch
import numpy as np
from collections import defaultdict

from data import (
    create_dataloaders,
    build_m_web,
    build_m_position_only,
    build_m_simple_structure,
    build_m_random,
)
from model import create_model
from train import train_model


# =========================================================================
# Configuration
# =========================================================================

CONFIG = {
    "num_samples": 3000,
    "batch_size": 128,
    "num_classes": 8,
    "vocab_size": 500,
    "max_seq_len": 80,
    "signature_size": 12,
    "center_bonus": 0.30,
    "support_bonus": 0.10,
    "desc_bonus": 0.02,
    "d_model": 128,
    "n_heads": 4,
    "n_layers": 2,
    "d_ff": 512,
    "epochs": 10,
    "lr": 1e-3,
    "device": "cpu",
}

SEEDS = [42, 123, 2024, 3407, 9999]

VARIANTS = [
    ("A_Transformer",     "none",   None,                      0.0),
    ("B_Position",        "pos",    build_m_position_only,     0.5),
    ("C_SimpleStructure", "simple", build_m_simple_structure,  0.5),
    ("D_SpiderWeb",       "full",   build_m_web,               0.5),
    ("E_RandomBias",      "full",   build_m_random,            0.5),
]

# Length grouping thresholds (based on actual data distribution)
LENGTH_GROUPS = {"short": (0, 50), "medium": (50, 65), "long": (65, 999)}


# =========================================================================
# Metrics (manual numpy implementation, no sklearn dependency)
# =========================================================================

def compute_metrics(y_true, y_pred, n_classes):
    """Compute multi-class accuracy, precision, recall, macro-F1, weighted-F1."""
    # Confusion-matrix style
    tp = np.zeros(n_classes)
    fp = np.zeros(n_classes)
    fn = np.zeros(n_classes)
    support = np.zeros(n_classes)

    for c in range(n_classes):
        tp[c] = np.sum((y_pred == c) & (y_true == c))
        fp[c] = np.sum((y_pred == c) & (y_true != c))
        fn[c] = np.sum((y_pred != c) & (y_true == c))
        support[c] = np.sum(y_true == c)

    # Per-class
    precision = np.zeros(n_classes)
    recall = np.zeros(n_classes)
    f1 = np.zeros(n_classes)
    for c in range(n_classes):
        precision[c] = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) > 0 else 0.0
        recall[c] = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) > 0 else 0.0
        f1[c] = 2 * precision[c] * recall[c] / (precision[c] + recall[c]) if (precision[c] + recall[c]) > 0 else 0.0

    accuracy = np.sum(tp) / len(y_true) if len(y_true) > 0 else 0.0
    macro_f1 = np.mean(f1)
    weighted_f1 = np.sum(f1 * support) / np.sum(support) if np.sum(support) > 0 else 0.0

    return {
        "accuracy": accuracy,
        "precision": np.mean(precision),
        "recall": np.mean(recall),
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class_f1": f1.tolist(),
    }


def compute_length_grouped(y_true, y_pred, seq_lens, n_classes, groups):
    """Compute metrics for each length group."""
    result = {}
    for group_name, (lo, hi) in groups.items():
        mask = (seq_lens >= lo) & (seq_lens < hi)
        if mask.sum() == 0:
            result[group_name] = {"accuracy": None, "macro_f1": None, "count": 0}
            continue
        m = compute_metrics(y_true[mask], y_pred[mask], n_classes)
        m["count"] = int(mask.sum())
        result[group_name] = m
    return result


# =========================================================================
# Main Experiment
# =========================================================================

def run_experiment(config):
    all_rows = []

    for seed_idx, seed in enumerate(SEEDS):
        print(f"\n{'=' * 60}")
        print(f"SEED {seed} ({seed_idx + 1}/{len(SEEDS)})")
        print(f"{'=' * 60}")

        torch.manual_seed(seed)

        # Create data with this seed
        train_loader, test_loader, ds = create_dataloaders(
            num_samples=config["num_samples"],
            batch_size=config["batch_size"],
            num_classes=config["num_classes"],
            vocab_size=config["vocab_size"],
            max_seq_len=config["max_seq_len"],
            seed=seed,
            signature_size=config["signature_size"],
            center_bonus=config["center_bonus"],
            support_bonus=config["support_bonus"],
            desc_bonus=config["desc_bonus"],
        )

        for var_name, bias_mode, build_fn, lambda_ in VARIANTS:
            print(f"\n  [{var_name}] mode={bias_mode}, lambda={lambda_}")

            # Reset torch seed for model init
            torch.manual_seed(seed)

            model = create_model(
                bias_mode=bias_mode,
                vocab_size=config["vocab_size"],
                d_model=config["d_model"],
                n_heads=config["n_heads"],
                n_layers=config["n_layers"],
                d_ff=config["d_ff"],
                n_classes=config["num_classes"],
                max_len=config["max_seq_len"],
            )

            history = train_model(
                model, train_loader, test_loader,
                epochs=config["epochs"],
                lr=config["lr"],
                lambda_=lambda_,
                build_bias_fn=build_fn,
                device=config["device"],
                verbose=True,
            )

            # Full-set metrics
            y_true = history["detail_labels"]
            y_pred = history["detail_preds"]
            seq_lens = history["detail_seq_lens"]
            n_classes = config["num_classes"]

            metrics = compute_metrics(y_true, y_pred, n_classes)

            # Length-grouped metrics
            lg = compute_length_grouped(y_true, y_pred, seq_lens, n_classes, LENGTH_GROUPS)

            row = {
                "seed": seed,
                "variant": var_name,
                "bias_mode": bias_mode,
                "lambda": lambda_,
                "accuracy": round(metrics["accuracy"], 6),
                "precision": round(metrics["precision"], 6),
                "recall": round(metrics["recall"], 6),
                "macro_f1": round(metrics["macro_f1"], 6),
                "weighted_f1": round(metrics["weighted_f1"], 6),
                "train_time_s": round(history["time"], 1),
                "best_epoch_acc": round(max(history["test_acc"]), 6),
                "final_epoch_acc": round(history["test_acc"][-1], 6),
            }
            for gname in LENGTH_GROUPS:
                row[f"acc_{gname}"] = round(lg[gname]["accuracy"], 6) if lg[gname]["accuracy"] is not None else None
                row[f"f1_{gname}"] = round(lg[gname]["macro_f1"], 6) if lg[gname]["macro_f1"] is not None else None
                row[f"count_{gname}"] = lg[gname]["count"]

            all_rows.append(row)

            print(f"    Acc={metrics['accuracy']:.4f}  "
                  f"MF1={metrics['macro_f1']:.4f}  "
                  f"WF1={metrics['weighted_f1']:.4f}  "
                  f"Time={history['time']:.0f}s")
            print(f"    Length: short={row.get('acc_short','-'):} "
                  f"med={row.get('acc_medium','-'):} "
                  f"long={row.get('acc_long','-'):}")

    return all_rows


# =========================================================================
# Stats & Output
# =========================================================================

def compute_stats(rows, metric="accuracy"):
    """Compute mean ± std per variant."""
    groups = defaultdict(list)
    for r in rows:
        groups[r["variant"]].append(r[metric])
    stats = {}
    for name in ["A_Transformer", "B_Position", "C_SimpleStructure", "D_SpiderWeb", "E_RandomBias"]:
        vals = groups[name]
        stats[name] = {"mean": np.mean(vals), "std": np.std(vals), "n": len(vals)}
    return stats


def print_stats_table(rows, metrics):
    """Print mean +/- std table for given metrics."""
    for metric in metrics:
        stats = compute_stats(rows, metric)
        print(f"\n{'=' * 70}")
        print(f"  {metric.upper()} (mean +/- std over {len(SEEDS)} seeds)")
        print(f"{'=' * 70}")
        order = ["A_Transformer", "B_Position", "C_SimpleStructure", "D_SpiderWeb", "E_RandomBias"]
        print(f"{'Variant':<22s} {'Mean':>10s} {'Std':>10s} {'Min':>10s} {'Max':>10s}")
        print("-" * 70)
        for name in order:
            s = stats[name]
            groups = defaultdict(list)
            for r in rows:
                groups[r["variant"]].append(r[metric])
            vals = [r[metric] for r in rows if r["variant"] == name]
            print(f"{name:<22s} {s['mean']:10.4f} {s['std']:10.4f} {min(vals):10.4f} {max(vals):10.4f}")

        # SpiderWeb vs Transformer
        sw = stats["D_SpiderWeb"]["mean"]
        tr = stats["A_Transformer"]["mean"]
        delta = sw - tr
        print(f"\n  SpiderWeb - Transformer = {delta:+.4f} ({delta * 100:+.2f}%)")
        rb = stats["E_RandomBias"]["mean"]
        print(f"  SpiderWeb - RandomBias  = {sw - rb:+.4f} ({(sw - rb) * 100:+.2f}%)")


# =========================================================================
# Save
# =========================================================================

def save_results(rows, config):
    base = os.path.dirname(os.path.abspath(__file__))

    # CSV
    csv_path = os.path.join(base, "experiment_results.csv")
    fieldnames = [
        "seed", "variant", "bias_mode", "lambda",
        "accuracy", "precision", "recall", "macro_f1", "weighted_f1",
        "train_time_s", "best_epoch_acc", "final_epoch_acc",
        "acc_short", "f1_short", "count_short",
        "acc_medium", "f1_medium", "count_medium",
        "acc_long", "f1_long", "count_long",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCSV saved: {csv_path}")

    # JSON (backward compat)
    json_path = os.path.join(base, "experiment_results.json")
    # Reformat for backward compatibility
    ablation_json = []
    for r in rows:
        ablation_json.append({
            "variant": r["variant"],
            "bias_mode": r["bias_mode"],
            "lambda": r["lambda"],
            "seed": r["seed"],
            "best_test_acc": r["best_epoch_acc"],
            "final_test_acc": r["final_epoch_acc"],
            "macro_f1": r["macro_f1"],
            "weighted_f1": r["weighted_f1"],
            "train_time_s": r["train_time_s"],
            "accuracy": r["accuracy"],
        })
    json_data = {
        "ablation": ablation_json,
        "config": {k: v for k, v in config.items() if not callable(v)},
        "seeds": SEEDS,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    print(f"JSON saved: {json_path}")


# =========================================================================
# Entry
# =========================================================================

if __name__ == "__main__":
    t0 = time.time()
    rows = run_experiment(CONFIG)

    # Print stats for key metrics
    print_stats_table(rows, ["accuracy", "macro_f1", "weighted_f1"])

    # Length-grouped stats
    print(f"\n{'=' * 70}")
    print("  LENGTH-GROUPED ACCURACY (mean over seeds)")
    print(f"{'=' * 70}")
    for gname in ["short", "medium", "long"]:
        groups = defaultdict(list)
        for r in rows:
            val = r.get(f"acc_{gname}")
            if val is not None:
                groups[r["variant"]].append(val)
        print(f"  [{gname}]")
        for name in ["A_Transformer", "B_Position", "C_SimpleStructure", "D_SpiderWeb", "E_RandomBias"]:
            vals = groups[name]
            if vals:
                print(f"    {name:<22s} {np.mean(vals):.4f} +/- {np.std(vals):.4f}")

    save_results(rows, CONFIG)
    print(f"\nTotal experiment time: {time.time() - t0:.0f}s")
