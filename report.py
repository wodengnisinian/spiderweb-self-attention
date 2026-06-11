# report.py -- SpiderWeb Paper-Ready Analysis Orchestrator
import os, sys
BASE=os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,BASE)

from analysis_core import (load_csv,print_summary,export_per_seed,export_length_grouped,export_paper_table,ORDER,gbv)
from analysis_viz import plot_acc_bars,plot_f1_bars,plot_length_grouped
from analysis_extras import run_inference_and_export,generate_paper_report

def main():
    # Determine output dir
    out_dir=os.path.join(BASE,"phase3_statistical_analysis")
    os.makedirs(out_dir,exist_ok=True)

    # Load data
    csv_path=os.path.join(BASE,"phase2_rigorous_experiment","experiment_results.csv")
    if not os.path.exists(csv_path):
        csv_path=os.path.join(BASE,"experiment_results.csv")
    if not os.path.exists(csv_path):
        print("ERROR: experiment_results.csv not found!"); sys.exit(1)
    rows=load_csv(csv_path)
    print(f"Loaded {len(rows)} rows")

    # 1. Console summary with effect size
    print_summary(rows)

    # 2. Export tables
    export_per_seed(rows,out_dir)
    export_length_grouped(rows,out_dir)
    export_paper_table(rows,out_dir)

    # 3. Plots
    plot_acc_bars(rows,out_dir)
    plot_f1_bars(rows,out_dir)
    plot_length_grouped(rows,out_dir)

    # 4. Confusion matrices + case studies + heatmaps (runs fresh inference)
    run_inference_and_export(rows,out_dir,csv_path)

    # 5. Paper-ready report
    generate_paper_report(rows,out_dir)

    print("\n"+"="*60)
    print("  DONE - paper_ready_experiment_report.md generated")
    print("="*60)

if __name__=="__main__":
    main()
