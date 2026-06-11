# analysis_core.py -- Statistics helpers for SpiderWeb
import csv, numpy as np, json, os
from collections import defaultdict
from scipy import stats

ORDER = ["A_Transformer","B_Position","C_SimpleStructure","D_SpiderWeb","E_RandomBias"]
METRICS = ["accuracy","precision","recall","macro_f1","weighted_f1"]
CP = [("D_SpiderWeb","A_Transformer","SWeb vs Transformer"),
      ("D_SpiderWeb","E_RandomBias","SWeb vs RandomBias"),
      ("D_SpiderWeb","C_SimpleStructure","SWeb vs SimpleStruct")]

def load_csv(path):
    rows=[]
    with open(path,"r",encoding="utf-8") as f:
        for r in csv.DictReader(f):
            row={}
            for k,v in r.items():
                try: row[k]=float(v)
                except: row[k]=v
            rows.append(row)
    return rows

def gbv(rows,m):
    g=defaultdict(list)
    for r in rows: g[r["variant"]].append(r[m])
    return g

def pv(rows,a,b,m):
    seeds=sorted(set(r["seed"] for r in rows))
    va,vb=[],[]
    for s in seeds:
        ra=[r for r in rows if r["seed"]==s and r["variant"]==a]
        rb=[r for r in rows if r["seed"]==s and r["variant"]==b]
        if ra and rb: va.append(ra[0][m]); vb.append(rb[0][m])
    return np.array(va),np.array(vb)

def cohens_d(va,vb):
    d=va-vb; sd=np.std(d,ddof=1)
    return np.mean(d)/sd if sd>0 else 0.0

def paired_t_test(rows,a,b,m):
    va,vb=pv(rows,a,b,m); n=len(va)
    t,p=stats.ttest_rel(va,vb); diff=np.mean(va-vb)
    sd=np.std(va-vb,ddof=1)
    ci=stats.t.interval(0.95,df=n-1,loc=diff,scale=sd/np.sqrt(n))
    return t,p,diff,ci,cohens_d(va,vb)

def wilcoxon_test(rows,a,b,m):
    va,vb=pv(rows,a,b,m)
    if np.allclose(va,vb): return 0.0,1.0
    return stats.wilcoxon(va,vb,zero_method="zsplit")

def print_summary(rows):
    print("\n"+"="*75)
    print("  ACCURACY (mean +/- std, 5 seeds)")
    print("="*75)
    groups=gbv(rows,"accuracy"); base=np.mean(groups["A_Transformer"])
    for name in ORDER:
        vals=np.array(groups[name]); m,s=np.mean(vals),np.std(vals)
        pp=(m-base)*100; rel=pp/(base*100)*100
        extra=""
        if name!="A_Transformer": extra=f" d={cohens_d(vals,np.array(groups['A_Transformer'])):.3f}"
        mr=" <<" if name=="D_SpiderWeb" else ""
        print(f"  {name:<22s} {m:.4f} +/- {s:.4f}  ({pp:+.2f} pp, rel {rel:+.2f}%){extra}{mr}")
    print()
    print("="*75)
    print("  PAIRED T-TESTS + EFFECT SIZE")
    print("="*75)
    for a,b,lab in CP:
        for m in ["accuracy","macro_f1"]:
            t,p,diff,ci,d=paired_t_test(rows,a,b,m)
            sig="***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "ns"
            print(f"  {lab:<30s} {m:<12s} diff={diff*100:+.2f}pp  t={t:.4f}  p={p:.4f} {sig}  d={d:.3f}")
    print()
    print("  WILCOXON:")
    for a,b,lab in CP:
        for m in ["accuracy","macro_f1"]:
            w,p=wilcoxon_test(rows,a,b,m)
            sig="***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "ns"
            print(f"  {lab:<30s} {m:<12s} W={w:.1f}  p={p:.4f} {sig}")

def export_per_seed(rows,out_dir):
    p=os.path.join(out_dir,"per_seed_raw_results.csv")
    with open(p,"w",newline="",encoding="utf-8") as f:
        w=csv.writer(f)
        w.writerow(["Seed","Variant","Accuracy","Precision","Recall","MacroF1","WeightedF1"])
        for r in sorted(rows,key=lambda x:(x["seed"],ORDER.index(x["variant"]) if x["variant"] in ORDER else 99)):
            w.writerow([int(r["seed"]),r["variant"],f"{r['accuracy']:.4f}",f"{r['precision']:.4f}",f"{r['recall']:.4f}",f"{r['macro_f1']:.4f}",f"{r['weighted_f1']:.4f}"])
    print(f"Per-seed: {p}")

def export_length_grouped(rows,out_dir):
    p=os.path.join(out_dir,"length_grouped_table.csv")
    with open(p,"w",newline="",encoding="utf-8") as f:
        w=csv.writer(f)
        w.writerow(["Group","Variant","AccMean","AccStd","F1Mean","F1Std","CountMean","DeltaPP_vs_A"])
        gnames=["short","medium","long"]
        base_acc={}
        for gn in gnames:
            vals=[r[f"acc_{gn}"] for r in rows if r["variant"]=="A_Transformer" and r.get(f"acc_{gn}") is not None]
            base_acc[gn]=np.mean(vals) if vals else 0
        for gn in gnames:
            for name in ORDER:
                accs=[r[f"acc_{gn}"] for r in rows if r["variant"]==name and r.get(f"acc_{gn}") is not None]
                f1s=[r[f"f1_{gn}"] for r in rows if r["variant"]==name and r.get(f"f1_{gn}") is not None]
                cts=[r[f"count_{gn}"] for r in rows if r["variant"]==name and r.get(f"count_{gn}") is not None]
                if accs:
                    delta=(np.mean(accs)-base_acc[gn])*100 if name!="A_Transformer" else 0
                    w.writerow([gn,name,f"{np.mean(accs):.4f}",f"{np.std(accs):.4f}",f"{np.mean(f1s):.4f}" if f1s else "",f"{np.std(f1s):.4f}" if f1s else "",f"{np.mean(cts):.0f}" if cts else "",f"{delta:+.2f}"])
    print(f"Length-grouped: {p}")

def export_paper_table(rows,out_dir):
    p=os.path.join(out_dir,"paper_table.md")
    groups=gbv(rows,"accuracy"); base=np.mean(groups["A_Transformer"]); bv=np.array(groups["A_Transformer"])
    lines=[]
    lines.append("### Table 1: Model Comparison (mean +/- std, 5 seeds)")
    lines.append("")
    lines.append("| Model | Accuracy | Precision | Recall | Macro-F1 | Abs. Imp. (pp) | Rel. Imp. (%) |")
    lines.append("|---|---|---|---|---|---|---|")
    labs=["A: Transformer","B: +Position","C: +Simple Struct.","D: +SpiderWeb","E: +Random Bias"]
    for i,name in enumerate(ORDER):
        parts=[labs[i]]
        for m in ["accuracy","precision","recall","macro_f1"]:
            vals=np.array(gbv(rows,m)[name])
            parts.append(f"{np.mean(vals):.4f} +/- {np.std(vals):.4f}")
        if name=="A_Transformer": parts.extend(["--","--"])
        else:
            vals=np.array(groups[name]); abs_imp=np.mean(vals)-base; rel_imp=abs_imp/base*100
            parts.append(f"{abs_imp*100:+.2f} pp"); parts.append(f"{rel_imp:+.2f}%")
        lines.append("| "+" | ".join(parts)+" |")
    lines.append("")
    lines.append("### Table 2: Statistical Tests (paired, 5 seeds)")
    lines.append("")
    lines.append("| Comparison | Metric | Mean Diff (pp) | 95% CI (pp) | t | p | Cohen's d | Wilcoxon p |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for a,b,lab in CP:
        for m in ["accuracy","macro_f1"]:
            t,pv2,diff,ci,d=paired_t_test(rows,a,b,m)
            w,pw=wilcoxon_test(rows,a,b,m)
            lines.append(f"| {lab} | {m} | {diff*100:+.2f} | [{ci[0]*100:+.2f},{ci[1]*100:+.2f}] | {t:.4f} | {pv2:.4f} | {d:.3f} | {pw:.4f} |")
    with open(p,"w",encoding="utf-8") as f: f.write("\n".join(lines))
    print(f"Paper table: {p}")
