#!/usr/bin/env python3
"""
Regenerate all 4 experiment figures from unified pipeline data.
===============================================================
Figures:
  1. fig4_anomaly_auprc_comparison.png — AUPRC bar chart (Telco, unified pipeline)
  2. fig6_stability_boxplot.png — F1 & AUPRC stability across 11 runs (Telco)
3. fig7_ctane_comparison.png — CTANE F1 across hyperparameters (Telco + Adult)
  4. fig_pr_curve.png — Precision-Recall curves (Telco, unified pipeline)

All data from:
  - experiments/results/unified/unified_results.json (Telco)
  - experiments/results/unified/adult_unified_results.json (Adult)
  - Re-computed Telco PR curve from cached LLM responses
"""
import json, os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.metrics import precision_recall_curve, average_precision_score
warnings.filterwarnings('ignore')

# ============================================================
# CONFIG
# ============================================================
BASE = "/sessions/69e790d3616697199023cd0a/workspace/mqw7dr6nxdtvfujn"
RESULTS_DIR = os.path.join(BASE, "experiments/results")
FIGURES_PATH = os.path.join(BASE, "experiments/results/figures")
os.makedirs(FIGURES_PATH, exist_ok=True)

# Font setup
plt.rcParams.update({
    'font.family': ['DejaVu Sans', 'Noto Sans CJK SC', 'WenQuanYi Micro Hei'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# Color palette
COLORS = {
    'hscv': '#2563eb',       # blue
    'individual': '#6b7280',  # gray
    'best': '#059669',       # green
    'eif': '#f59e0b',        # amber
    'copod': '#ef4444',      # red
    'fusion': '#8b5cf6',     # purple
    'cfdminer': '#dc2626',   # dark red
    'llm_cfd': '#2563eb',    # blue
}

# ============================================================
# LOAD DATA
# ============================================================
def load_json(path):
    with open(path) as f:
        return json.load(f)

telco = load_json(os.path.join(RESULTS_DIR, "unified/unified_results.json"))
adult = load_json(os.path.join(RESULTS_DIR, "unified/adult_unified_results.json"))

print(f"Telco: {len(telco['individual_runs']['per_run'])} runs, HSCV AUPRC={telco['hscv_fixed']['auprc']:.4f}")
print(f"Adult: {len(adult['individual_runs']['per_run'])} runs, HSCV AUPRC={adult['hscv_fixed']['auprc']:.4f}")

# ============================================================
# FIGURE 4: AUPRC Comparison (Telco, Unified Pipeline)
# ============================================================
def generate_fig4_auprc_comparison():
    """Bar chart comparing AUPRC across all methods on Telco dataset."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    methods = [
        ('LLM-CFD\n(median)', telco['individual_runs']['auprc_median'], COLORS['individual']),
        ('LLM-CFD\n(best)', telco['individual_runs']['auprc_best'], COLORS['best']),
        ('LLM-CFD\n+HSCV', telco['hscv_fixed']['auprc'], COLORS['hscv']),
        ('EIF', telco['baselines']['eif_auprc'], COLORS['eif']),
        ('COPOD', telco['baselines']['copod_auprc'], COLORS['copod']),
        ('HSCV+COPOD\n(fusion)', telco['fusion']['hscv_copod_auprc'], COLORS['fusion']),
        ('LLM-CFD(best)\n+COPOD', telco['fusion']['llm_cfd_best_copod_auprc'], COLORS['fusion']),
    ]
    
    labels = [m[0] for m in methods]
    values = [m[1] for m in methods]
    colors = [m[2] for m in methods]
    
    bars = ax.bar(range(len(methods)), values, color=colors, edgecolor='white', linewidth=0.8, width=0.65)
    
    # Add value labels on top of bars
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Add median line
    ax.axhline(y=telco['individual_runs']['auprc_median'], color=COLORS['individual'],
               linestyle='--', alpha=0.5, label=f'LLM-CFD median ({telco["individual_runs"]["auprc_median"]:.3f})')
    
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('AUPRC')
    ax.set_title('Anomaly Detection AUPRC Comparison (Telco Churn, Unified Pipeline)', fontsize=13, fontweight='bold')
    ax.set_ylim(0, max(values) * 1.15)
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    for fmt in ['png', 'svg']:
        path = os.path.join(FIGURES_PATH, f'fig4_anomaly_auprc_comparison.{fmt}')
        fig.savefig(path, dpi=300 if fmt == 'png' else None)
        print(f"  Saved: {path}")
    plt.close(fig)

# ============================================================
# FIGURE 6: Stability Boxplot (Telco, 11 runs)
# ============================================================
def generate_fig6_stability_boxplot():
    """Box plot showing F1 and AUPRC stability across 11 runs."""
    per_run = telco['individual_runs']['per_run']
    f1_vals = [r['f1'] for r in per_run]
    auprc_vals = [r['auprc'] for r in per_run]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))
    
    # F1 boxplot
    bp1 = ax1.boxplot([f1_vals], positions=[1], widths=0.5, patch_artist=True,
                      boxprops=dict(facecolor=COLORS['hscv'], alpha=0.3),
                      medianprops=dict(color=COLORS['hscv'], linewidth=2),
                      whiskerprops=dict(color=COLORS['hscv']),
                      capprops=dict(color=COLORS['hscv']),
                      flierprops=dict(marker='o', markerfacecolor=COLORS['hscv'], markersize=5))
    
    # Add individual points
    x_jitter = np.random.normal(1, 0.04, len(f1_vals))
    ax1.scatter(x_jitter, f1_vals, color=COLORS['hscv'], alpha=0.6, s=30, zorder=3)
    
    # Add HSCV line
    ax1.axhline(y=telco['hscv_fixed']['f1'], color=COLORS['best'], linestyle='--',
                label=f'LLM-CFD+HSCV ({telco["hscv_fixed"]["f1"]:.3f})')
    # Add best line
    f1_best = max(f1_vals)
    ax1.axhline(y=f1_best, color=COLORS['llm_cfd'], linestyle=':', alpha=0.7,
                label=f'LLM-CFD best ({f1_best:.3f})')
    
    ax1.set_ylabel('F1 Score')
    ax1.set_title(f'F1 Stability (n={len(f1_vals)} runs)', fontsize=12, fontweight='bold')
    ax1.set_xticks([1])
    ax1.set_xticklabels(['LLM-CFD'])
    ax1.legend(fontsize=9)
    ax1.grid(axis='y', alpha=0.3)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Add stats text
    f1_median = np.median(f1_vals)
    f1_iqr = np.percentile(f1_vals, 75) - np.percentile(f1_vals, 25)
    ax1.text(1.3, f1_median, f'median={f1_median:.3f}\nIQR={f1_iqr:.3f}', fontsize=8, va='center')
    
    # AUPRC boxplot
    bp2 = ax2.boxplot([auprc_vals], positions=[1], widths=0.5, patch_artist=True,
                      boxprops=dict(facecolor=COLORS['eif'], alpha=0.3),
                      medianprops=dict(color=COLORS['eif'], linewidth=2),
                      whiskerprops=dict(color=COLORS['eif']),
                      capprops=dict(color=COLORS['eif']),
                      flierprops=dict(marker='o', markerfacecolor=COLORS['eif'], markersize=5))
    
    x_jitter2 = np.random.normal(1, 0.04, len(auprc_vals))
    ax2.scatter(x_jitter2, auprc_vals, color=COLORS['eif'], alpha=0.6, s=30, zorder=3)
    
    ax2.axhline(y=telco['hscv_fixed']['auprc'], color=COLORS['best'], linestyle='--',
                label=f'LLM-CFD+HSCV ({telco["hscv_fixed"]["auprc"]:.3f})')
    # Add best line
    auprc_best_val = max(auprc_vals)
    ax2.axhline(y=auprc_best_val, color=COLORS['llm_cfd'], linestyle=':', alpha=0.7,
                label=f'LLM-CFD best ({auprc_best_val:.3f})')
    
    ax2.set_ylabel('AUPRC')
    ax2.set_title(f'AUPRC Stability (n={len(auprc_vals)} runs)', fontsize=12, fontweight='bold')
    ax2.set_xticks([1])
    ax2.set_xticklabels(['LLM-CFD'])
    ax2.legend(fontsize=9)
    ax2.grid(axis='y', alpha=0.3)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    auprc_median = np.median(auprc_vals)
    auprc_iqr = np.percentile(auprc_vals, 75) - np.percentile(auprc_vals, 25)
    ax2.text(1.3, auprc_median, f'median={auprc_median:.3f}\nIQR={auprc_iqr:.3f}', fontsize=8, va='center')
    
    plt.suptitle('LLM-CFD Output Stability Across Independent Runs (Telco Churn)', 
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    for fmt in ['png', 'svg']:
        path = os.path.join(FIGURES_PATH, f'fig6_stability_boxplot.{fmt}')
        fig.savefig(path, dpi=300 if fmt == 'png' else None)
        print(f"  Saved: {path}")
    plt.close(fig)

# ============================================================
# FIGURE 7: CFDMiner Comparison (Telco + Adult)
# ============================================================
def generate_fig7_ctane_comparison():
    """Bar chart comparing CTANE F1 across hyperparameters for both datasets."""
    # Load CTANE data from separate results file
    ctane_path = os.path.join(RESULTS_DIR, "ctane/ctane_experiment_results.json")
    ctane_data = load_json(ctane_path) if os.path.exists(ctane_path) else {}

    # Telco CTANE data
    telco_ctane = ctane_data.get('telco', {})
    telco_configs = telco_ctane.get('all_configs', [])
    telco_f1 = [c['f1'] for c in telco_configs]
    telco_labels = [f"sup={c['min_support']:.3f}\nconf={c['min_confidence']:.2f}" for c in telco_configs]
    telco_best_idx = np.argmax(telco_f1) if telco_f1 else 0

    # Adult CTANE data
    adult_ctane = ctane_data.get('adult', {})
    adult_configs = adult_ctane.get('all_configs', [])
    adult_f1 = [c['f1'] for c in adult_configs]
    adult_best_idx = np.argmax(adult_f1) if adult_f1 else 0
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    
    x = np.arange(len(telco_f1))
    width = 0.6
    
    # Telco
    colors_telco = [COLORS['cfdminer'] if i == telco_best_idx else '#fca5a5' for i in range(len(telco_f1))]
    ax1.bar(x, telco_f1, width, color=colors_telco, edgecolor='white', linewidth=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(telco_labels, fontsize=7, rotation=45, ha='right')
    ax1.set_ylabel('F1 Score')
    ax1.set_title(f'Telco Churn (GT=15 rules)\nBest: F1={max(telco_f1):.3f}', fontsize=11, fontweight='bold')
    ax1.axhline(y=telco['hscv_fixed']['f1'], color=COLORS['hscv'], linestyle='--',
                label=f'LLM-CFD+HSCV F1={telco["hscv_fixed"]["f1"]:.3f}')
    telco_llm_best_f1 = max(r['f1'] for r in telco['individual_runs']['per_run'])
    ax1.axhline(y=telco_llm_best_f1, color=COLORS['llm_cfd'], linestyle=':', alpha=0.7,
                label=f'LLM-CFD best F1={telco_llm_best_f1:.3f}')
    ax1.legend(fontsize=9)
    ax1.grid(axis='y', alpha=0.3)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.annotate(f'{max(telco_f1):.3f}', xy=(telco_best_idx, max(telco_f1)),
                xytext=(telco_best_idx, max(telco_f1) + 0.005),
                ha='center', fontsize=8, fontweight='bold', color=COLORS['cfdminer'])
    
    # Adult
    colors_adult = [COLORS['cfdminer'] if i == adult_best_idx else '#fca5a5' for i in range(len(adult_f1))]
    adult_labels = [f"sup={c['min_support']:.3f}\nconf={c['min_confidence']:.2f}" for c in adult_configs]
    ax2.bar(x, adult_f1, width, color=colors_adult, edgecolor='white', linewidth=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(adult_labels, fontsize=7, rotation=45, ha='right')
    ax2.set_ylabel('F1 Score')
    ax2.set_title(f'UCI Adult (GT=12 rules)\nBest: F1={max(adult_f1):.3f}', fontsize=11, fontweight='bold')
    ax2.axhline(y=adult['hscv_fixed']['f1'], color=COLORS['hscv'], linestyle='--',
                label=f'LLM-CFD+HSCV F1={adult["hscv_fixed"]["f1"]:.3f}')
    adult_llm_best_f1 = max(r['f1'] for r in adult['individual_runs']['per_run'])
    ax2.axhline(y=adult_llm_best_f1, color=COLORS['llm_cfd'], linestyle=':', alpha=0.7,
                label=f'LLM-CFD best F1={adult_llm_best_f1:.3f}')
    ax2.legend(fontsize=9)
    ax2.grid(axis='y', alpha=0.3)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.annotate(f'{max(adult_f1):.3f}', xy=(adult_best_idx, max(adult_f1)),
                xytext=(adult_best_idx, max(adult_f1) + 0.003),
                ha='center', fontsize=8, fontweight='bold', color=COLORS['cfdminer'])
    
    plt.suptitle('CTANE F1 Score Across Hyperparameter Configurations',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    for fmt in ['png', 'svg']:
        path = os.path.join(FIGURES_PATH, f'fig7_ctane_comparison.{fmt}')
        fig.savefig(path, dpi=300 if fmt == 'png' else None)
        print(f"  Saved: {path}")
    plt.close(fig)

# ============================================================
# FIGURE 8: PR Curve (Telco, Unified Pipeline)
# ============================================================
def generate_fig_pr_curve():
    """Precision-Recall curves for HSCV, EIF, COPOD, and fusion on Telco dataset."""
    # Import from unified_experiment.py (the authoritative unified pipeline)
    sys.path.insert(0, os.path.join(BASE, "experiments/src"))
    from unified_experiment import (
        load_cached_response, parse_cfds_robust,
        hybrid_self_consistency_voting, run_eif, run_copod,
        unified_pipeline_score, normalize_scores,
        RANDOM_SEED,
    )
    from self_consistency_voting_v2 import (
        load_data, inject_anomalies, validate_cfds,
        GROUND_TRUTH_CFDS, CACHE_DIR,
    )
    
    print("  Loading Telco data and recomputing unified pipeline scores...")
    df = load_data()
    df_clean = df.copy()
    df_anom, labels = inject_anomalies(df, seed=RANDOM_SEED)
    valid_columns = set(df.columns)
    
    # Load cached LLM responses using unified_experiment's robust parser
    cache_files = sorted([f for f in os.listdir(CACHE_DIR) if f.endswith('.json')])
    print(f"  Found {len(cache_files)} cache files")
    all_runs_cfds = []
    for cf in cache_files:
        response = load_cached_response(os.path.join(CACHE_DIR, cf))
        cfds = parse_cfds_robust(response, valid_columns)
        all_runs_cfds.append(cfds)
        print(f"    {cf}: {len(cfds)} CFDs")
    
    # HSCV using unified_experiment's voting
    voted_cfds = hybrid_self_consistency_voting(all_runs_cfds)
    voted_cfds = [c for c in voted_cfds if isinstance(c, dict)]
    validated_hscv = validate_cfds(df_clean, [c.copy() for c in voted_cfds])
    hscv_scores, _ = unified_pipeline_score(df_anom, validated_hscv, df_clean)
    hscv_auprc = average_precision_score(labels, hscv_scores)
    
    # EIF (using unified_experiment's run_eif)
    eif_scores = run_eif(df_anom)
    eif_auprc = average_precision_score(labels, eif_scores)
    
    # COPOD (using unified_experiment's run_copod)
    copod_scores = run_copod(df_anom)
    copod_auprc = average_precision_score(labels, copod_scores)
    
    # Fusion: HSCV + COPOD
    norm_hscv = normalize_scores(hscv_scores)
    norm_copod = normalize_scores(copod_scores)
    fusion_scores = (norm_hscv + norm_copod) / 2.0
    fusion_auprc = average_precision_score(labels, fusion_scores)
    
    # Best individual run + COPOD
    best_auprc = -1
    best_scores = None
    for cfds in all_runs_cfds:
        run_cfds = [c for c in cfds if isinstance(c, dict)]
        validated = validate_cfds(df_clean, [c.copy() for c in run_cfds])
        if len(validated) == 0:
            continue
        scores, _ = unified_pipeline_score(df_anom, validated, df_clean)
        auprc = average_precision_score(labels, scores)
        if auprc > best_auprc:
            best_auprc = auprc
            best_scores = scores.copy()
    
    norm_best = normalize_scores(best_scores) if best_scores is not None else np.zeros_like(labels, dtype=float)
    fusion_best_copod = (norm_best + norm_copod) / 2.0
    fusion_best_auprc = average_precision_score(labels, fusion_best_copod)
    
    print(f"  HSCV AUPRC={hscv_auprc:.4f}, EIF={eif_auprc:.4f}, COPOD={copod_auprc:.4f}")
    print(f"  Fusion(HSCV+COPOD)={fusion_auprc:.4f}, Fusion(Best+COPOD)={fusion_best_auprc:.4f}")
    
    # Compute PR curves
    prec_llm_cfd_best, rec_llm_cfd_best, _ = precision_recall_curve(labels, best_scores)
    prec_hscv, rec_hscv, _ = precision_recall_curve(labels, hscv_scores)
    prec_eif, rec_eif, _ = precision_recall_curve(labels, normalize_scores(eif_scores))
    prec_copod, rec_copod, _ = precision_recall_curve(labels, normalize_scores(copod_scores))
    prec_fusion, rec_fusion, _ = precision_recall_curve(labels, fusion_scores)
    prec_fusion_best, rec_fusion_best, _ = precision_recall_curve(labels, fusion_best_copod)
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot(rec_llm_cfd_best, prec_llm_cfd_best, color=COLORS['best'], linewidth=2.5,
            label=f'LLM-CFD best (AUPRC={best_auprc:.3f})')
    ax.plot(rec_hscv, prec_hscv, color=COLORS['hscv'], linewidth=2.5,
            label=f'LLM-CFD+HSCV (AUPRC={hscv_auprc:.3f})')
    ax.plot(rec_fusion_best, prec_fusion_best, color=COLORS['fusion'], linewidth=2.5, linestyle='-',
            label=f'LLM-CFD(best)+COPOD (AUPRC={fusion_best_auprc:.3f})')
    ax.plot(rec_fusion, prec_fusion, color=COLORS['best'], linewidth=2, linestyle='--',
            label=f'HSCV+COPOD (AUPRC={fusion_auprc:.3f})')
    ax.plot(rec_eif, prec_eif, color=COLORS['eif'], linewidth=1.5, linestyle='-',
            label=f'EIF (AUPRC={eif_auprc:.3f})')
    ax.plot(rec_copod, prec_copod, color=COLORS['copod'], linewidth=1.5, linestyle='-',
            label=f'COPOD (AUPRC={copod_auprc:.3f})')
    
    # Baseline (random)
    anomaly_rate = labels.mean()
    ax.axhline(y=anomaly_rate, color='gray', linestyle=':', alpha=0.5,
               label=f'Random baseline ({anomaly_rate:.3f})')
    
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curves (Telco Churn, Unified Pipeline)', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    for fmt in ['png', 'svg']:
        path = os.path.join(FIGURES_PATH, f'fig_pr_curve.{fmt}')
        fig.savefig(path, dpi=300 if fmt == 'png' else None)
        print(f"  Saved: {path}")
    plt.close(fig)

# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 80)
    print("REGENERATING ALL 4 EXPERIMENT FIGURES (Unified Pipeline)")
    print("=" * 80)
    
    print("\n[1/4] Fig4: AUPRC Comparison (Telco)")
    generate_fig4_auprc_comparison()
    
    print("\n[2/4] Fig6: Stability Boxplot (Telco, 11 runs)")
    generate_fig6_stability_boxplot()
    
    print("\n[3/4] Fig7: CTANE Comparison (Telco + Adult)")
    generate_fig7_ctane_comparison()
    
    print("\n[4/4] Fig8: PR Curve (Telco, recomputed)")
    generate_fig_pr_curve()
    
    print("\n" + "=" * 80)
    print("ALL 4 FIGURES GENERATED SUCCESSFULLY")
    print(f"Output directory: {FIGURES_PATH}")
    print("=" * 80)

if __name__ == "__main__":
    main()
