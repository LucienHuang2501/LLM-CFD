#!/usr/bin/env python3
"""
Regenerate all 4 experiment figures with Nature journal visual standards.
=========================================================================
Color scheme (Nature-style low-saturation):
  - LLM-CFD methods: blue gradient
  - Statistical baselines: muted earth tones
  - Fusion methods: distinct colors
  - CTANE: muted red

Figures:
  1. fig4_anomaly_auprc_comparison — AUPRC bar chart (Telco)
  2. fig6_stability_boxplot — F1 & AUPRC stability (Telco, 11 runs)
  3. fig7_ctane_comparison — CTANE F1 across hyperparams (Telco + Adult)
  4. fig_pr_curve — Precision-Recall curves (Telco)
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
# NATURE MANDATORY rcParams
# ============================================================
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.size'] = 8
plt.rcParams['axes.labelsize'] = 9
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['xtick.labelsize'] = 7.5
plt.rcParams['ytick.labelsize'] = 7.5
plt.rcParams['legend.fontsize'] = 7.5
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['axes.linewidth'] = 0.5
plt.rcParams['xtick.major.width'] = 0.5
plt.rcParams['ytick.major.width'] = 0.5
plt.rcParams['xtick.major.size'] = 3
plt.rcParams['ytick.major.size'] = 3
plt.rcParams['lines.linewidth'] = 1.0

# ============================================================
# CONFIG
# ============================================================
BASE = "/sessions/69e790d3616697199023cd0a/workspace/mqw7dr6nxdtvfujn"
RESULTS_DIR = os.path.join(BASE, "experiments/results")
FIGURES_PATH = os.path.join(BASE, "experiments/results/figures")
os.makedirs(FIGURES_PATH, exist_ok=True)

# Nature-style low-saturation palette
# LLM-CFD methods: blue gradient
# Statistical baselines: muted earth tones
# Fusion methods: distinct colors
NATURE_COLORS = {
    'llm_individual': '#8fa8c4',  # light blue-gray (individual median)
    'llm_best':      '#5a7fa6',   # medium blue (best individual)
    'llm_hscv':      '#4a7a74',   # teal (HSCV — primary method)
    'eif':           '#9e7b3f',   # muted gold (statistical baseline)
    'copod':         '#a67060',   # muted rust (statistical baseline)
    'fusion':        '#7a6a8a',   # muted purple (fusion)
    'fusion_best':   '#6a5a7a',   # darker purple (best+fusion)
    'ctane_best':    '#a05050',   # muted red (CTANE best)
    'ctane_other':   '#c4a0a0',   # light muted red (CTANE other configs)
    'median_line':   '#888888',   # gray (reference lines)
    'random':        '#bbbbbb',   # light gray (random baseline)
}

# Shape encoding for color-blind safety
MARKERS = {
    'llm_individual': 'o',
    'llm_best': 's',
    'llm_hscv': 'D',
    'eif': '^',
    'copod': 'v',
    'fusion': 'P',
    'fusion_best': 'X',
}

# ============================================================
# LOAD DATA
# ============================================================
def load_json(path):
    with open(path) as f:
        return json.load(f)

telco = load_json(os.path.join(RESULTS_DIR, "unified/unified_results.json"))
adult = load_json(os.path.join(RESULTS_DIR, "unified/adult_unified_results.json"))
ctane_data = load_json(os.path.join(RESULTS_DIR, "ctane/ctane_experiment_results.json"))

print(f"Telco: {len(telco['individual_runs']['per_run'])} runs, HSCV AUPRC={telco['hscv_fixed']['auprc']:.4f}")
print(f"Adult: {len(adult['individual_runs']['per_run'])} runs, HSCV AUPRC={adult['hscv_fixed']['auprc']:.4f}")

# ============================================================
# FIGURE 4: AUPRC Comparison (Telco)
# ============================================================
def generate_fig4_auprc_comparison():
    """Bar chart comparing AUPRC across all methods on Telco dataset."""
    fig, ax = plt.subplots(figsize=(7, 4.5))

    # Simplified: CoT+fewshot is deterministic, so median=best=HSCV
    methods = [
        ('LLM-CFD\n(CoT+fewshot)',  telco['individual_runs']['auprc_median'], NATURE_COLORS['llm_hscv']),
        ('EIF',                      telco['baselines']['eif_auprc'],          NATURE_COLORS['eif']),
        ('COPOD',                    telco['baselines']['copod_auprc'],        NATURE_COLORS['copod']),
        ('LLM-CFD\n+COPOD',         telco['fusion']['hscv_copod_auprc'],      NATURE_COLORS['fusion']),
    ]

    labels = [m[0] for m in methods]
    values = [m[1] for m in methods]
    colors = [m[2] for m in methods]

    x = np.arange(len(methods))
    bars = ax.bar(x, values, color=colors, edgecolor='white', linewidth=0.4, width=0.6)

    # Hatching for statistical baselines (color-blind safety dual-encoding)
    for i, (name, _, _) in enumerate(methods):
        if name in ('EIF', 'COPOD'):
            bars[i].set_hatch('//')
        elif 'COPOD' in name:
            bars[i].set_hatch('xx')

    # Value labels
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.015,
                f'{val:.3f}', ha='center', va='bottom', fontsize=7, fontweight='bold',
                color='#333333')

    # Reference line: median
    ax.axhline(y=telco['individual_runs']['auprc_median'], color=NATURE_COLORS['median_line'],
               linestyle='--', linewidth=0.6, alpha=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel('AUPRC')
    ax.set_ylim(0, max(values) * 1.18)
    ax.grid(axis='y', alpha=0.2, linewidth=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Legend for method categories
    legend_elements = [
        Patch(facecolor=NATURE_COLORS['llm_hscv'], edgecolor='white', label='LLM-CFD methods'),
        Patch(facecolor=NATURE_COLORS['eif'], edgecolor='white', hatch='//', label='Statistical baselines'),
        Patch(facecolor=NATURE_COLORS['fusion'], edgecolor='white', hatch='xx', label='Fusion methods'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', framealpha=0.9,
              edgecolor='#cccccc', fancybox=False)

    plt.tight_layout()
    for fmt in ['png', 'svg']:
        path = os.path.join(FIGURES_PATH, f'fig4_anomaly_auprc_comparison.{fmt}')
        fig.savefig(path, dpi=300 if fmt == 'png' else None)
        print(f"  Saved: {path}")
    plt.close(fig)

# ============================================================
# FIGURE 6: Stability Boxplot (Telco, 15 runs)
# ============================================================
def generate_fig6_stability_boxplot():
    """Box plot showing F1 and AUPRC stability across runs."""
    per_run = telco['individual_runs']['per_run']
    f1_vals = [r['f1'] for r in per_run]
    auprc_vals = [r['auprc'] for r in per_run]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

    # --- F1 panel ---
    bp1 = ax1.boxplot([f1_vals], positions=[1], widths=0.4, patch_artist=True,
                      boxprops=dict(facecolor=NATURE_COLORS['llm_best'], alpha=0.25,
                                    edgecolor=NATURE_COLORS['llm_best'], linewidth=0.6),
                      medianprops=dict(color=NATURE_COLORS['llm_best'], linewidth=1.2),
                      whiskerprops=dict(color=NATURE_COLORS['llm_best'], linewidth=0.6),
                      capprops=dict(color=NATURE_COLORS['llm_best'], linewidth=0.6),
                      flierprops=dict(marker='o', markerfacecolor=NATURE_COLORS['llm_best'],
                                      markersize=3, markeredgecolor='none', alpha=0.7))

    x_jitter = np.random.normal(1, 0.03, len(f1_vals))
    ax1.scatter(x_jitter, f1_vals, color=NATURE_COLORS['llm_best'], alpha=0.6, s=12, zorder=3,
                edgecolors='white', linewidths=0.3)

    # HSCV line
    ax1.axhline(y=telco['hscv_fixed']['f1'], color=NATURE_COLORS['llm_hscv'], linestyle='--',
                linewidth=0.8, label=f'HSCV ({telco["hscv_fixed"]["f1"]:.3f})')
    # Best line
    f1_best = max(f1_vals)
    ax1.axhline(y=f1_best, color=NATURE_COLORS['llm_individual'], linestyle=':',
                linewidth=0.7, alpha=0.8, label=f'Best ({f1_best:.3f})')

    ax1.set_ylabel('F1 Score')
    ax1.set_title(f'F1 Stability (n={len(f1_vals)})', fontweight='bold')
    ax1.set_xticks([1])
    ax1.set_xticklabels(['LLM-CFD'])
    ax1.legend(fontsize=7, framealpha=0.9, edgecolor='#cccccc', fancybox=False)
    ax1.grid(axis='y', alpha=0.2, linewidth=0.4)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    f1_median = np.median(f1_vals)
    f1_iqr = np.percentile(f1_vals, 75) - np.percentile(f1_vals, 25)
    ax1.text(1.25, f1_median, f'median={f1_median:.3f}\nIQR={f1_iqr:.3f}', fontsize=6.5,
             va='center', color='#555555')

    # --- AUPRC panel ---
    bp2 = ax2.boxplot([auprc_vals], positions=[1], widths=0.4, patch_artist=True,
                      boxprops=dict(facecolor=NATURE_COLORS['eif'], alpha=0.25,
                                    edgecolor=NATURE_COLORS['eif'], linewidth=0.6),
                      medianprops=dict(color=NATURE_COLORS['eif'], linewidth=1.2),
                      whiskerprops=dict(color=NATURE_COLORS['eif'], linewidth=0.6),
                      capprops=dict(color=NATURE_COLORS['eif'], linewidth=0.6),
                      flierprops=dict(marker='o', markerfacecolor=NATURE_COLORS['eif'],
                                      markersize=3, markeredgecolor='none', alpha=0.7))

    x_jitter2 = np.random.normal(1, 0.03, len(auprc_vals))
    ax2.scatter(x_jitter2, auprc_vals, color=NATURE_COLORS['eif'], alpha=0.6, s=12, zorder=3,
                edgecolors='white', linewidths=0.3)

    ax2.axhline(y=telco['hscv_fixed']['auprc'], color=NATURE_COLORS['llm_hscv'], linestyle='--',
                linewidth=0.8, label=f'HSCV ({telco["hscv_fixed"]["auprc"]:.3f})')
    auprc_best_val = max(auprc_vals)
    ax2.axhline(y=auprc_best_val, color=NATURE_COLORS['llm_individual'], linestyle=':',
                linewidth=0.7, alpha=0.8, label=f'Best ({auprc_best_val:.3f})')

    ax2.set_ylabel('AUPRC')
    ax2.set_title(f'AUPRC Stability (n={len(auprc_vals)})', fontweight='bold')
    ax2.set_xticks([1])
    ax2.set_xticklabels(['LLM-CFD'])
    ax2.legend(fontsize=7, framealpha=0.9, edgecolor='#cccccc', fancybox=False)
    ax2.grid(axis='y', alpha=0.2, linewidth=0.4)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    auprc_median = np.median(auprc_vals)
    auprc_iqr = np.percentile(auprc_vals, 75) - np.percentile(auprc_vals, 25)
    ax2.text(1.25, auprc_median, f'median={auprc_median:.3f}\nIQR={auprc_iqr:.3f}', fontsize=6.5,
             va='center', color='#555555')

    plt.tight_layout()
    for fmt in ['png', 'svg']:
        path = os.path.join(FIGURES_PATH, f'fig6_stability_boxplot.{fmt}')
        fig.savefig(path, dpi=300 if fmt == 'png' else None)
        print(f"  Saved: {path}")
    plt.close(fig)

# ============================================================
# FIGURE 7: CTANE Comparison (Telco + Adult)
# ============================================================
def generate_fig7_ctane_comparison():
    """Bar chart comparing CTANE F1 across hyperparameters for both datasets."""
    telco_ctane = ctane_data.get('telco', {})
    telco_configs = telco_ctane.get('all_configs', [])
    telco_f1 = [c['f1'] for c in telco_configs]
    telco_labels = [f"sup={c['min_support']:.3f}\nconf={c['min_confidence']:.2f}" for c in telco_configs]
    telco_best_idx = int(np.argmax(telco_f1)) if telco_f1 else 0

    adult_ctane = ctane_data.get('adult', {})
    adult_configs = adult_ctane.get('all_configs', [])
    adult_f1 = [c['f1'] for c in adult_configs]
    adult_best_idx = int(np.argmax(adult_f1)) if adult_f1 else 0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))

    x_t = np.arange(len(telco_f1))
    width = 0.55

    # Telco panel
    colors_t = [NATURE_COLORS['ctane_best'] if i == telco_best_idx else NATURE_COLORS['ctane_other']
                for i in range(len(telco_f1))]
    bars_t = ax1.bar(x_t, telco_f1, width, color=colors_t, edgecolor='white', linewidth=0.4)
    # Hatching for non-best bars
    for i, bar in enumerate(bars_t):
        if i != telco_best_idx:
            bar.set_hatch('///')
    ax1.set_xticks(x_t)
    ax1.set_xticklabels(telco_labels, fontsize=5.5, rotation=45, ha='right')
    ax1.set_ylabel('F1 Score')
    ax1.set_title(f'Telco Churn (GT=15)\nBest CTANE: F1={max(telco_f1):.3f}', fontweight='bold', fontsize=8.5)

    # Reference lines
    ax1.axhline(y=telco['hscv_fixed']['f1'], color=NATURE_COLORS['llm_hscv'], linestyle='--',
                linewidth=0.7, label=f'LLM-CFD+HSCV ({telco["hscv_fixed"]["f1"]:.3f})')
    telco_llm_best_f1 = max(r['f1'] for r in telco['individual_runs']['per_run'])
    ax1.axhline(y=telco_llm_best_f1, color=NATURE_COLORS['llm_best'], linestyle=':',
                linewidth=0.6, alpha=0.8, label=f'LLM-CFD best ({telco_llm_best_f1:.3f})')
    ax1.legend(fontsize=6.5, framealpha=0.9, edgecolor='#cccccc', fancybox=False)
    ax1.grid(axis='y', alpha=0.2, linewidth=0.4)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.annotate(f'{max(telco_f1):.3f}', xy=(telco_best_idx, max(telco_f1)),
                 xytext=(telco_best_idx, max(telco_f1) + 0.008),
                 ha='center', fontsize=6.5, fontweight='bold', color=NATURE_COLORS['ctane_best'])

    # Adult panel
    x_a = np.arange(len(adult_f1))
    colors_a = [NATURE_COLORS['ctane_best'] if i == adult_best_idx else NATURE_COLORS['ctane_other']
                for i in range(len(adult_f1))]
    bars_a = ax2.bar(x_a, adult_f1, width, color=colors_a, edgecolor='white', linewidth=0.4)
    for i, bar in enumerate(bars_a):
        if i != adult_best_idx:
            bar.set_hatch('///')
    adult_labels = [f"sup={c['min_support']:.3f}\nconf={c['min_confidence']:.2f}" for c in adult_configs]
    ax2.set_xticks(x_a)
    ax2.set_xticklabels(adult_labels, fontsize=5.5, rotation=45, ha='right')
    ax2.set_ylabel('F1 Score')
    ax2.set_title(f'UCI Adult (GT=12)\nBest CTANE: F1={max(adult_f1):.3f}', fontweight='bold', fontsize=8.5)

    ax2.axhline(y=adult['hscv_fixed']['f1'], color=NATURE_COLORS['llm_hscv'], linestyle='--',
                linewidth=0.7, label=f'LLM-CFD+HSCV ({adult["hscv_fixed"]["f1"]:.3f})')
    adult_llm_best_f1 = max(r['f1'] for r in adult['individual_runs']['per_run'])
    ax2.axhline(y=adult_llm_best_f1, color=NATURE_COLORS['llm_best'], linestyle=':',
                linewidth=0.6, alpha=0.8, label=f'LLM-CFD best ({adult_llm_best_f1:.3f})')
    ax2.legend(fontsize=6.5, framealpha=0.9, edgecolor='#cccccc', fancybox=False)
    ax2.grid(axis='y', alpha=0.2, linewidth=0.4)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.annotate(f'{max(adult_f1):.3f}', xy=(adult_best_idx, max(adult_f1)),
                 xytext=(adult_best_idx, max(adult_f1) + 0.005),
                 ha='center', fontsize=6.5, fontweight='bold', color=NATURE_COLORS['ctane_best'])

    plt.tight_layout()
    for fmt in ['png', 'svg']:
        path = os.path.join(FIGURES_PATH, f'fig7_ctane_comparison.{fmt}')
        fig.savefig(path, dpi=300 if fmt == 'png' else None)
        print(f"  Saved: {path}")
    plt.close(fig)

# ============================================================
# FIGURE: PR Curve (Telco)
# ============================================================
def generate_fig_pr_curve():
    """Precision-Recall curves for HSCV, EIF, COPOD, and fusion on Telco dataset."""
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

    # Override CACHE_DIR to use cot_fewshot cache (deterministic runs)
    COT_CACHE_DIR = os.path.join(RESULTS_DIR, "supplementary/cache_cot")

    print("  Loading Telco data and recomputing unified pipeline scores...")
    df = load_data()
    df_clean = df.copy()
    df_anom, labels = inject_anomalies(df, seed=RANDOM_SEED)
    valid_columns = set(df.columns)

    cache_files = sorted([f for f in os.listdir(COT_CACHE_DIR) if f.endswith('.json')])
    print(f"  Found {len(cache_files)} cache files")
    all_runs_cfds = []
    for cf in cache_files:
        response = load_cached_response(os.path.join(COT_CACHE_DIR, cf))
        cfds = parse_cfds_robust(response, valid_columns)
        all_runs_cfds.append(cfds)
        print(f"    {cf}: {len(cfds)} CFDs")

    # HSCV
    voted_cfds = hybrid_self_consistency_voting(all_runs_cfds)
    voted_cfds = [c for c in voted_cfds if isinstance(c, dict)]
    validated_hscv = validate_cfds(df_clean, [c.copy() for c in voted_cfds])
    hscv_scores, _ = unified_pipeline_score(df_anom, validated_hscv, df_clean)
    hscv_auprc = average_precision_score(labels, hscv_scores)

    # EIF
    eif_scores = run_eif(df_anom)
    eif_auprc = average_precision_score(labels, eif_scores)

    # COPOD
    copod_scores = run_copod(df_anom)
    copod_auprc = average_precision_score(labels, copod_scores)

    # Fusion: HSCV + COPOD
    norm_hscv = normalize_scores(hscv_scores)
    norm_copod = normalize_scores(copod_scores)
    fusion_scores = (norm_hscv + norm_copod) / 2.0
    fusion_auprc = average_precision_score(labels, fusion_scores)

    # Best individual + COPOD
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

    # Add tiny jitter to break score ties (prevents artificial step-bends in PR curve)
    rng = np.random.RandomState(42)
    def jitter_scores(scores):
        s = np.asarray(scores, dtype=float)
        s = np.nan_to_num(s, nan=0.0)
        unique_count = len(np.unique(s))
        if unique_count < len(s) * 0.1:  # Many ties → add jitter
            max_val = max(s.max(), 1e-8)
            s = s + rng.uniform(0, max_val * 0.001, size=len(s))
        return s

    best_j = jitter_scores(best_scores)
    hscv_j = jitter_scores(hscv_scores)
    eif_j = jitter_scores(normalize_scores(eif_scores))
    copod_j = jitter_scores(normalize_scores(copod_scores))
    fusion_j = jitter_scores(fusion_scores)
    fb_j = jitter_scores(fusion_best_copod)

    # PR curves (jittered for smooth rendering)
    prec_best, rec_best, _ = precision_recall_curve(labels, best_j)
    prec_hscv, rec_hscv, _ = precision_recall_curve(labels, hscv_j)
    prec_eif, rec_eif, _ = precision_recall_curve(labels, eif_j)
    prec_copod, rec_copod, _ = precision_recall_curve(labels, copod_j)
    prec_fusion, rec_fusion, _ = precision_recall_curve(labels, fusion_j)
    prec_fb, rec_fb, _ = precision_recall_curve(labels, fb_j)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    # LLM-CFD methods (blue gradient)
    ax.plot(rec_best, prec_best, color=NATURE_COLORS['llm_best'], linewidth=1.2,
            label=f'LLM-CFD best (AUPRC={best_auprc:.3f})')
    ax.plot(rec_hscv, prec_hscv, color=NATURE_COLORS['llm_hscv'], linewidth=1.5,
            label=f'LLM-CFD+HSCV (AUPRC={hscv_auprc:.3f})')

    # Statistical baselines (earth tones, thinner)
    ax.plot(rec_eif, prec_eif, color=NATURE_COLORS['eif'], linewidth=1.0, linestyle='--',
            label=f'EIF (AUPRC={eif_auprc:.3f})')
    ax.plot(rec_copod, prec_copod, color=NATURE_COLORS['copod'], linewidth=1.0, linestyle='--',
            label=f'COPOD (AUPRC={copod_auprc:.3f})')

    # Fusion methods (purple)
    ax.plot(rec_fb, prec_fb, color=NATURE_COLORS['fusion_best'], linewidth=1.2, linestyle='-',
            label=f'LLM-CFD(best)+COPOD ({fusion_best_auprc:.3f})')
    ax.plot(rec_fusion, prec_fusion, color=NATURE_COLORS['fusion'], linewidth=1.0, linestyle=':',
            label=f'HSCV+COPOD ({fusion_auprc:.3f})')

    # Random baseline
    anomaly_rate = labels.mean()
    ax.axhline(y=anomaly_rate, color=NATURE_COLORS['random'], linestyle=':', linewidth=0.6,
               label=f'Random ({anomaly_rate:.3f})')

    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    # Legend outside plot area to avoid covering curves
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=7,
              framealpha=0.95, edgecolor='#cccccc', fancybox=False,
              borderpad=0.6, labelspacing=0.4)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.15, linewidth=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    for fmt in ['png', 'svg']:
        path = os.path.join(FIGURES_PATH, f'fig_pr_curve.{fmt}')
        fig.savefig(path, dpi=300 if fmt == 'png' else None)
        print(f"  Saved: {path}")
    plt.close(fig)

# ============================================================
# QA SUMMARY
# ============================================================
def print_qa_summary():
    print("\n" + "=" * 60)
    print("NATURE-STYLE FIGURE QA SUMMARY")
    print("=" * 60)
    print(f"Font family: {plt.rcParams['font.family']}")
    print(f"SVG fonttype: {plt.rcParams['svg.fonttype']}")
    print(f"Base font size: {plt.rcParams['font.size']}pt")
    print(f"Color palette (Nature low-saturation):")
    for name, color in NATURE_COLORS.items():
        print(f"  {name:20s}: {color}")
    print(f"Output directory: {FIGURES_PATH}")
    print(f"Formats: PNG (300 DPI) + SVG (vector)")
    print("=" * 60)

# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 60)
    print("REGENERATING FIGURES WITH NATURE JOURNAL STYLE")
    print("=" * 60)

    print("\n[1/4] Fig4: AUPRC Comparison (Telco)")
    generate_fig4_auprc_comparison()

    print("\n[2/4] Fig6: Stability Boxplot (Telco)")
    generate_fig6_stability_boxplot()

    print("\n[3/4] Fig7: CTANE Comparison (Telco + Adult)")
    generate_fig7_ctane_comparison()

    print("\n[4/4] PR Curve (Telco, recomputed)")
    generate_fig_pr_curve()

    print_qa_summary()
    print("\nALL 4 FIGURES GENERATED SUCCESSFULLY")

if __name__ == "__main__":
    main()
