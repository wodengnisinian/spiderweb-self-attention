# analysis_viz.py -- All plotting functions for SpiderWeb paper
import os, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from analysis_core import ORDER, gbv

SHORT=["A\nTransformer","B\n+Position","C\n+Simple","D\nSpiderWeb","E\nRandom"]
COLORS=["#3498db","#2ecc71","#f39c12","#e74c3c","#95a5a6"]

def plot_acc_bars(rows,out_dir):
    groups=gbv(rows,"accuracy")
    means=[np.mean(groups[n]) for n in ORDER]
    stds=[np.std(groups[n]) for n in ORDER]
    base_acc=means[0]
    fig,ax=plt.subplots(figsize=(13,5))
    bars=ax.bar(SHORT,means,yerr=stds,color=COLORS,capsize=8,width=0.55,edgecolor="white")
    for i,(bar,m,s) in enumerate(zip(bars,means,stds)):
        label=f"{m:.4f}"
        if i>0:
            pp=(m-base_acc)*100; rel=pp/(base_acc*100)*100
            label+=f"\n(+{pp:+.2f} pp, {rel:+.2f}%)"
        ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+s+0.003,label,ha="center",va="bottom",fontsize=9,fontweight="bold")
    sw=means[3]
    ax.annotate(f"SWeb - Transformer = {(sw-base_acc)*100:+.2f} pp",xy=(3,sw),xytext=(3,sw+0.03),fontsize=11,color="#e74c3c",fontweight="bold",ha="center",arrowprops=dict(arrowstyle="->",color="#e74c3c"))
    ax.set_ylabel("Accuracy"); ax.set_title("Accuracy (mean +/- std, 5 seeds)")
    ax.grid(True,alpha=0.3,axis="y"); ax.set_ylim(bottom=min(means)-0.04)
    plt.tight_layout(); p=os.path.join(out_dir,"accuracy_bars.png"); fig.savefig(p); plt.close(fig)
    print(f"Saved: {p}")

def plot_f1_bars(rows,out_dir):
    groups=gbv(rows,"macro_f1")
    means=[np.mean(groups[n]) for n in ORDER]
    stds=[np.std(groups[n]) for n in ORDER]
    base=means[0]
    fig,ax=plt.subplots(figsize=(13,5))
    bars=ax.bar(SHORT,means,yerr=stds,color=COLORS,capsize=8,width=0.55,edgecolor="white")
    for i,(bar,m,s) in enumerate(zip(bars,means,stds)):
        label=f"{m:.4f}"
        if i>0:
            pp=(m-base)*100; rel=pp/(base*100)*100
            label+=f"\n(+{pp:+.2f} pp, {rel:+.2f}%)"
        ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+s+0.003,label,ha="center",va="bottom",fontsize=9,fontweight="bold")
    sw=means[3]
    ax.annotate(f"SWeb - Transformer = {(sw-base)*100:+.2f} pp",xy=(3,sw),xytext=(3,sw+0.03),fontsize=11,color="#e74c3c",fontweight="bold",ha="center",arrowprops=dict(arrowstyle="->",color="#e74c3c"))
    ax.set_ylabel("Macro-F1"); ax.set_title("Macro-F1 (mean +/- std, 5 seeds)")
    ax.grid(True,alpha=0.3,axis="y"); ax.set_ylim(bottom=min(means)-0.04)
    plt.tight_layout(); p=os.path.join(out_dir,"f1_bars.png"); fig.savefig(p); plt.close(fig)
    print(f"Saved: {p}")

def plot_length_grouped(rows,out_dir):
    gnames=["short","medium","long"]; glabels=["Short (< 50)","Medium (50-65)","Long (> 65)"]
    fig,axes=plt.subplots(1,3,figsize=(19,5),sharey=True)
    for gi,(gn,gl) in enumerate(zip(gnames,glabels)):
        ax=axes[gi]; means,stds=[],[]
        for name in ORDER:
            vals=[r[f"acc_{gn}"] for r in rows if r["variant"]==name and r.get(f"acc_{gn}") is not None]
            means.append(np.mean(vals) if vals else 0); stds.append(np.std(vals) if vals else 0)
        ax.bar(SHORT,means,yerr=stds,color=COLORS,capsize=6,width=0.55,edgecolor="white")
        ax.set_title(gl); ax.grid(True,alpha=0.3,axis="y"); ax.tick_params(axis="x",rotation=0)
        if gi==0: ax.set_ylabel("Accuracy")
        btm=max(0,min(m for m in means if m>0)-0.08); ax.set_ylim(bottom=btm,top=max(means)+0.1)
    fig.suptitle("Accuracy by Text Length",fontsize=14,fontweight="bold")
    plt.tight_layout(); p=os.path.join(out_dir,"length_grouped.png"); fig.savefig(p); plt.close(fig)
    print(f"Saved: {p}")

def plot_confusion(y_true,y_pred,n_classes,title,out_dir,filename):
    cm=np.zeros((n_classes,n_classes),dtype=int)
    for t,p in zip(y_true,y_pred): cm[t,p]+=1
    fig,ax=plt.subplots(figsize=(7,6))
    sns.heatmap(cm,annot=True,fmt="d",cmap="Blues",ax=ax,xticklabels=range(n_classes),yticklabels=range(n_classes))
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    plt.tight_layout(); p=os.path.join(out_dir,filename); fig.savefig(p); plt.close(fig)
    print(f"Saved: {p}")

def plot_attn_heatmap(attn,title,out_dir,filename):
    if attn is None: print("No attn"); return
    sample=attn[0].mean(axis=0)
    fig,ax=plt.subplots(figsize=(8,7))
    sns.heatmap(sample,cmap="YlOrRd",ax=ax,cbar_kws={"label":"Attention"})
    ax.set_xlabel("Key"); ax.set_ylabel("Query"); ax.set_title(title)
    plt.tight_layout(); p=os.path.join(out_dir,filename); fig.savefig(p); plt.close(fig)
    print(f"Saved: {p}")

def plot_mweb_heatmap(mweb,title,out_dir,filename):
    if mweb is None: print("No M_web"); return
    fig,ax=plt.subplots(figsize=(8,7))
    sns.heatmap(mweb[0],cmap="RdBu_r",center=0,ax=ax,cbar_kws={"label":"Bias"})
    ax.set_xlabel("Key"); ax.set_ylabel("Query"); ax.set_title(title)
    plt.tight_layout(); p=os.path.join(out_dir,filename); fig.savefig(p); plt.close(fig)
    print(f"Saved: {p}")
