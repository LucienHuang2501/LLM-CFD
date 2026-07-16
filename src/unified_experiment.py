#!/usr/bin/env python3
"""
Unified Experiment: All methods under the SAME pipeline.
=========================================================
Runs ALL methods (individual LLM-CFD runs, HSCV, EIF, COPOD, fusions,
CFDMiner-BL) under identical conditions:

  - Anomaly injection: v2's inject_anomalies (semantic: dependency destroy
    10% + 4sigma range 5% + negative tenure logic 3%)
  - Scoring: gradient scoring via compute_violation_matrix(df, cfds, df_clean)
  - Rule processing: reduce_redundancy (group by dependent_attribute, take max)
  - Aggregation: score_topk_mean (top-3 mean)
  - Post-processing: apply_p99_clip

Usage:
    cd experiments
    python3 src/unified_experiment.py
"""
import json
import os
import sys
import re
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import LabelEncoder
from scipy import stats as sp_stats

warnings.filterwarnings('ignore')

# ============================================================
# PATH SETUP — import from sibling modules
# ============================================================
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Imports from self_consistency_voting_v2.py
from self_consistency_voting_v2 import (
    inject_anomalies,
    compute_violation_matrix,
    reduce_redundancy,
    score_topk_mean,
    apply_p99_clip,
    evaluate_condition,
    evaluate_dependency,
    validate_cfds,
    _compute_conditional_prob,
    cfd_key,
    cfd_key_with_type,
    GROUND_TRUTH_CFDS,
    TYPE_MAP,
    CATEGORICAL_TYPES,
    SUPPORT_THRESHOLD,
    CONFIDENCE_THRESHOLDS,
)

# Override CACHE_DIR to use cot_fewshot cache (deterministic runs)
CACHE_DIR = "/sessions/69e790d3616697199023cd0a/workspace/mqw7dr6nxdtvfujn/experiments/results/supplementary/cache_cot"

# Imports from experiment.py
from experiment import build_llm_prompt, parse_cfd_response

# load_data is not defined in experiment.py; fall back to v2
try:
    from experiment import load_data
except ImportError:
    from self_consistency_voting_v2 import load_data

# Imports from cfdminer_hyperparam_sweep.py
from cfdminer_hyperparam_sweep import cfdminer_baseline


# ============================================================
# CONSTANTS
# ============================================================
RANDOM_SEED = 42
RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'results', 'unified'
)


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def evaluate_discovery(discovered, ground_truth):
    """Evaluate CFD discovery quality: Precision, Recall, F1.

    Matching is based on (dependent_attribute, condition_attributes, type_category)
    using the TYPE_MAP normalization.
    """
    gt_keys = set(cfd_key(c) for c in ground_truth)
    disc_keys = set(cfd_key(c) for c in discovered)
    tp = len(gt_keys & disc_keys)
    fp = len(disc_keys - gt_keys)
    fn = len(gt_keys - disc_keys)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return {'precision': precision, 'recall': recall, 'f1': f1,
            'tp': tp, 'fp': fp, 'fn': fn}


def normalize_scores(scores):
    """Min-max normalize scores to [0, 1]."""
    mn, mx = scores.min(), scores.max()
    if mx - mn < 1e-8:
        return np.zeros_like(scores)
    return (scores - mn) / (mx - mn + 1e-8)


# ============================================================
# CACHE LOADING & PARSING (with truncation fallback)
# ============================================================

def load_cached_response(cache_file):
    """Load and clean a cached LLM response from file.

    Cache files may be:
      - A JSON string (double-quoted, escaped) containing the raw LLM response
      - A JSON object with a 'response' key
      - Raw text of the LLM response
    """
    with open(cache_file) as f:
        raw = f.read()

    # Try JSON decode first (file might be a JSON string or object)
    try:
        decoded = json.loads(raw)
        if isinstance(decoded, str):
            return decoded
        elif isinstance(decoded, dict) and 'response' in decoded:
            return decoded['response']
    except (json.JSONDecodeError, ValueError):
        pass

    # Manual cleanup for quoted strings
    raw = raw.strip()
    if raw.startswith('"'):
        raw = raw[1:]
    if raw.endswith('"'):
        raw = raw[:-1]
    raw = raw.replace('\\n', '\n').replace('\\"', '"')

    # Remove markdown code fences
    if raw.startswith('```json'):
        raw = raw[7:]
    elif raw.startswith('```'):
        raw = raw[3:]
    if raw.endswith('```'):
        raw = raw[:-3]

    return raw.strip()


def parse_cfds_robust(response, valid_columns):
    """Parse CFDs from LLM response with truncation fallback.

    1. First tries parse_cfd_response from experiment.py (standard parser).
    2. If that fails (e.g., truncated JSON), falls back to brace-depth
       tracking to extract complete top-level JSON objects.
    """
    # Step 1: Standard parser from experiment.py
    cfds = parse_cfd_response(response, valid_columns)
    if cfds:
        return cfds

    # Step 2: Fallback — brace-depth tracking for truncated JSON
    # (same approach as cross_llm_experiment.py's parse_cfds)
    json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
    else:
        json_match = re.search(r'\[\s*\{.*', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
        else:
            json_str = response

    objects = []
    depth = 0
    start = None
    for i, c in enumerate(json_str):
        if c == '{':
            if depth == 0:
                start = i
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0 and start is not None:
                objects.append(json_str[start:i + 1])
                start = None

    cfds = []
    for obj_str in objects:
        try:
            cfd = json.loads(obj_str)
            if 'type' in cfd and 'dependent_attribute' in cfd:
                if cfd['dependent_attribute'] in valid_columns:
                    cfds.append(cfd)
        except (json.JSONDecodeError, ValueError):
            pass

    return cfds


# ============================================================
# COPOD (Copula-Based Outlier Detection — from scratch)
# ============================================================

def run_copod(df_anom):
    """COPOD: Copula-Based Outlier Detection (implemented from paper).

    For each data point, compute left-tail and right-tail probabilities
    across all dimensions, then combine using skewness-corrected aggregation.

    Reference: Li et al., "COPOD: Copula-Based Outlier Detection", ICDM 2020.
    Also references experiment.py's run_copod implementation.

    Parameters
    ----------
    df_anom : pd.DataFrame
        Anomalous dataframe (same df_anom from inject_anomalies).

    Returns
    -------
    np.ndarray
        Anomaly scores (higher = more anomalous). NOT normalized —
        normalization is handled by the caller when needed for fusion.
    """
    # Label encode categorical columns
    df_enc = df_anom.copy()
    for col in df_enc.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df_enc[col] = le.fit_transform(df_enc[col].astype(str))

    X = df_enc.values.astype(float)
    n, d = X.shape

    # Step 1: Compute empirical CDF (left-tail) and 1-CDF (right-tail)
    #         for each dimension using fractional ranking.
    U_l = np.zeros((n, d))
    U_r = np.zeros((n, d))

    for j in range(d):
        col = X[:, j]
        # Fractional ranking handles ties properly
        ranks = sp_stats.rankdata(col, method='average')
        U_l[:, j] = ranks / (n + 1)    # Left-tail ECDF: P(X <= x)
        U_r[:, j] = 1.0 - U_l[:, j]     # Right-tail ECDF: P(X >= x)

    # Step 2: Compute negative log-probability for each tail
    #         (prevent log(0) with clipping)
    eps = 1e-10
    U_l = np.clip(U_l, eps, 1 - eps)
    U_r = np.clip(U_r, eps, 1 - eps)

    left_tail = -np.log(U_l)    # -log(P_left)
    right_tail = -np.log(U_r)   # -log(P_right)

    # Step 3: Skewness-corrected aggregation
    # For each dimension, select the tail based on the skewness direction:
    #   - Left-skewed (skewness < 0): use left tail (extreme low values are anomalous)
    #   - Right-skewed or symmetric (skewness >= 0): use right tail
    skewness = np.array([sp_stats.skew(X[:, j]) for j in range(d)])

    scores = np.zeros(n)
    for j in range(d):
        if skewness[j] < 0:
            scores += left_tail[:, j]
        else:
            scores += right_tail[:, j]

    return scores


# ============================================================
# EIF BASELINE (Extended Isolation Forest)
# ============================================================

def run_eif(df_anom):
    """Extended Isolation Forest baseline on the same df_anom.

    Uses sklearn's IsolationForest (which implements extended/random-split
    isolation). Returns raw anomaly scores (higher = more anomalous).
    """
    df_enc = df_anom.copy()
    for col in df_enc.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df_enc[col] = le.fit_transform(df_enc[col].astype(str))

    iforest = IsolationForest(
        n_estimators=200,
        contamination=0.18,
        random_state=RANDOM_SEED,
    )
    iforest.fit(df_enc)
    # score_samples returns negative anomaly scores; negate so higher = anomalous
    return -iforest.score_samples(df_enc)


# ============================================================
# UNIFIED PIPELINE
# ============================================================

def unified_pipeline_score(df_anom, validated_cfds, df_clean):
    """Unified scoring pipeline used by ALL CFD-based methods.

    Steps:
      1. Gradient violation matrix: compute_violation_matrix(df, cfds, df_clean)
      2. Redundancy reduction: reduce_redundancy (group by dep attr, take max)
      3. Aggregation: score_topk_mean (top-3 mean)
      4. Post-processing: apply_p99_clip

    Returns
    -------
    scores : np.ndarray
        Anomaly scores after p99 clipping.
    n_groups : int
        Number of rule groups after redundancy reduction.
    """
    V = compute_violation_matrix(df_anom, validated_cfds, df_clean=df_clean)
    V_r, group_cfds = reduce_redundancy(V, validated_cfds)
    scores = apply_p99_clip(score_topk_mean(V_r, group_cfds))
    return scores, len(group_cfds)


# ============================================================
# HSCV (Hybrid Self-Consistency Voting)
# ============================================================

def hybrid_self_consistency_voting(all_runs_cfds):
    """Aggregate multiple LLM runs using hybrid voting.

    - Categorical types (fd/enum/logic): majority vote (>= ceil(n_runs * 0.5))
    - Numerical/structural types (range/consistency): union (>= 1 run)

    This follows the logic from cross_llm_experiment.py lines 560-580.

    Parameters
    ----------
    all_runs_cfds : list of list of dict
        CFD lists from each LLM run.

    Returns
    -------
    list of dict
        Voted CFDs (best representative from each surviving key).
    """
    n_runs = len(all_runs_cfds)
    cat_min_votes = int(np.ceil(n_runs * 0.5))

    # Collect all CFDs by key (preserving original type)
    key_to_info = {}
    for run_idx, cfds in enumerate(all_runs_cfds):
        seen = set()
        for cfd in cfds:
            key = cfd_key_with_type(cfd)
            if key in seen:
                continue
            seen.add(key)
            if key not in key_to_info:
                key_to_info[key] = {
                    'cfds': [],
                    'count': 0,
                    'orig_type': cfd.get('type', 'fd'),
                }
            key_to_info[key]['cfds'].append(cfd)
            key_to_info[key]['count'] += 1

    # Apply hybrid selection
    voted_cfds = []
    for key, info in key_to_info.items():
        is_cat = info['orig_type'] in CATEGORICAL_TYPES
        if (is_cat and info['count'] >= cat_min_votes) or (not is_cat and info['count'] >= 1):
            best = max(info['cfds'], key=lambda c:
                       {'high': 3, 'medium': 2, 'low': 1}.get(
                           c.get('confidence_estimate', 'low'), 0))
            voted_cfds.append(best)

    return voted_cfds


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 80)
    print("UNIFIED EXPERIMENT: All methods under the SAME pipeline")
    print("  Anomaly injection: semantic (dependency destroy 10% + 4sigma range 5% + neg tenure 3%)")
    print("  Scoring: gradient (1-P(dep|condition)) + redundancy reduction + top3_mean + p99_clip")
    print("=" * 80)

    # ─── Load data ───
    df = load_data()
    df_clean = df.copy()
    df_anom, labels = inject_anomalies(df, seed=RANDOM_SEED)
    valid_columns = set(df.columns)
    print(f"\nData: {df.shape}, Anomalies: {int(labels.sum())} ({labels.mean() * 100:.1f}%)")

    # ─── Load cached LLM responses ───
    cache_files = sorted([f for f in os.listdir(CACHE_DIR) if f.endswith('.json')])
    print(f"Found {len(cache_files)} cache files in CACHE_DIR")

    all_runs_cfds = []  # List of CFD lists (one per run)
    for cf in cache_files:
        response = load_cached_response(os.path.join(CACHE_DIR, cf))
        cfds = parse_cfds_robust(response, valid_columns)
        all_runs_cfds.append(cfds)
        print(f"  {cf}: {len(cfds)} CFDs parsed")

    n_runs = len(all_runs_cfds)

    # ─── Compute baselines (EIF, COPOD) on the SAME df_anom ───
    print("\n" + "-" * 80)
    print("Computing baselines (EIF, COPOD) on same df_anom...")
    eif_scores = run_eif(df_anom)
    copod_scores = run_copod(df_anom)
    auprc_eif = average_precision_score(labels, eif_scores)
    auprc_copod = average_precision_score(labels, copod_scores)
    print(f"  EIF AUPRC:   {auprc_eif:.4f}")
    print(f"  COPOD AUPRC: {auprc_copod:.4f}")

    # ================================================================
    # (a) Individual LLM-CFD runs (unified pipeline)
    # ================================================================
    print("\n" + "=" * 80)
    print("(a) Individual LLM-CFD runs (unified pipeline)")
    print("=" * 80)

    per_run = []
    per_run_auprc = []
    per_run_f1 = []
    best_auprc = -1.0
    best_run_scores = None
    best_run_idx = -1

    for i, cfds in enumerate(all_runs_cfds):
        # Validate CFDs (use copies so originals are preserved for HSCV)
        validated = validate_cfds(df_clean, [c.copy() for c in cfds])
        metrics = evaluate_discovery(validated, GROUND_TRUTH_CFDS)

        # Unified pipeline scoring
        scores, n_groups = unified_pipeline_score(df_anom, validated, df_clean)
        # Handle NaN from empty/zero-validated runs
        scores = np.nan_to_num(scores, nan=0.0)
        auprc = average_precision_score(labels, scores)

        has_consistency = bool(any(c.get('type') == 'consistency' for c in validated))

        per_run.append({
            'n_cfd': len(cfds),
            'n_validated': len(validated),
            'n_groups': n_groups,
            'f1': float(metrics['f1']),
            'auprc': float(auprc),
            'has_consistency': has_consistency,
        })
        per_run_auprc.append(auprc)
        per_run_f1.append(metrics['f1'])

        if auprc > best_auprc:
            best_auprc = auprc
            best_run_scores = scores.copy()
            best_run_idx = i

        print(f"  Run {i + 1:2d}: n_cfd={len(cfds):2d} n_val={len(validated):2d} "
              f"n_grp={n_groups:2d} F1={metrics['f1']:.3f} AUPRC={auprc:.4f} "
              f"{'[cons]' if has_consistency else ''}")

    f1_median = float(np.median(per_run_f1))
    f1_iqr = float(np.percentile(per_run_f1, 75) - np.percentile(per_run_f1, 25))
    auprc_median = float(np.median(per_run_auprc))
    auprc_iqr = float(np.percentile(per_run_auprc, 75) - np.percentile(per_run_auprc, 25))
    auprc_best = float(max(per_run_auprc))

    print(f"\n  F1:    median={f1_median:.3f} IQR={f1_iqr:.3f}")
    print(f"  AUPRC: median={auprc_median:.4f} IQR={auprc_iqr:.4f} best={auprc_best:.4f} (Run {best_run_idx + 1})")
    print(f"  Runs with consistency: {sum(1 for r in per_run if r['has_consistency'])}/{n_runs}")

    # ================================================================
    # (b) HSCV fixed config (cat>=50%, top3_mean)
    # ================================================================
    print("\n" + "=" * 80)
    print("(b) HSCV fixed config (cat>=50%, top3_mean)")
    print("=" * 80)

    voted_cfds = hybrid_self_consistency_voting(all_runs_cfds)
    print(f"  Voted CFDs: {len(voted_cfds)}")

    validated_hscv = validate_cfds(df_clean, [c.copy() for c in voted_cfds])
    metrics_hscv = evaluate_discovery(validated_hscv, GROUND_TRUTH_CFDS)

    hscv_scores, hscv_n_groups = unified_pipeline_score(df_anom, validated_hscv, df_clean)
    hscv_scores = np.nan_to_num(hscv_scores, nan=0.0)
    hscv_auprc = average_precision_score(labels, hscv_scores)
    hscv_has_consistency = bool(any(c.get('type') == 'consistency' for c in validated_hscv))

    print(f"  Validated: {len(validated_hscv)} rules, {hscv_n_groups} groups")
    print(f"  F1={metrics_hscv['f1']:.3f} AUPRC={hscv_auprc:.4f} "
          f"{'[cons]' if hscv_has_consistency else ''}")

    # ================================================================
    # (c) & (d) Baselines (already computed above)
    # ================================================================
    print("\n" + "=" * 80)
    print("(c) EIF baseline  &  (d) COPOD baseline")
    print("=" * 80)
    print(f"  EIF AUPRC:   {auprc_eif:.4f}")
    print(f"  COPOD AUPRC: {auprc_copod:.4f}")

    # ================================================================
    # (e) LLM-CFD best + COPOD fusion
    # ================================================================
    print("\n" + "=" * 80)
    print("(e) LLM-CFD best + COPOD fusion")
    print("=" * 80)

    norm_best_llm = normalize_scores(np.nan_to_num(best_run_scores, nan=0.0))
    norm_copod = normalize_scores(copod_scores)
    fusion_llm_copod = (norm_best_llm + norm_copod) / 2.0
    auprc_fusion_llm_copod = average_precision_score(labels, fusion_llm_copod)
    print(f"  Best LLM-CFD (Run {best_run_idx + 1}) AUPRC: {best_auprc:.4f}")
    print(f"  COPOD AUPRC:  {auprc_copod:.4f}")
    print(f"  Fusion AUPRC: {auprc_fusion_llm_copod:.4f}")

    # ================================================================
    # (f) HSCV + COPOD fusion
    # ================================================================
    print("\n" + "=" * 80)
    print("(f) HSCV + COPOD fusion")
    print("=" * 80)

    norm_hscv = normalize_scores(hscv_scores)
    fusion_hscv_copod = (norm_hscv + norm_copod) / 2.0
    auprc_fusion_hscv_copod = average_precision_score(labels, fusion_hscv_copod)
    print(f"  HSCV AUPRC:   {hscv_auprc:.4f}")
    print(f"  COPOD AUPRC:  {auprc_copod:.4f}")
    print(f"  Fusion AUPRC: {auprc_fusion_hscv_copod:.4f}")

    # ================================================================
    # (g) CFDMiner-BL (discovery only, no anomaly detection)
    # ================================================================
    print("\n" + "=" * 80)
    print("(g) CFDMiner-BL (discovery only, no anomaly detection)")
    print("=" * 80)

    cfdminer_cfds = cfdminer_baseline(df_clean, min_support=0.005, min_confidence=0.95)
    cfdminer_metrics = evaluate_discovery(cfdminer_cfds, GROUND_TRUTH_CFDS)
    print(f"  Candidates: {len(cfdminer_cfds)}")
    print(f"  F1={cfdminer_metrics['f1']:.3f} P={cfdminer_metrics['precision']:.3f} R={cfdminer_metrics['recall']:.3f}")

    # ================================================================
    # SAVE RESULTS
    # ================================================================
    os.makedirs(RESULTS_DIR, exist_ok=True)

    output = {
        "description": "Unified pipeline: all methods under same anomaly injection + gradient scoring + redundancy reduction + top3_mean",
        "anomaly_injection": "semantic (dependency destroy 10% + 4sigma range 5% + negative tenure logic 3%)",
        "scoring": "gradient (1-P(dep|condition)) + redundancy reduction + top3_mean + p99_clip",
        "individual_runs": {
            "f1_median": f1_median,
            "f1_iqr": f1_iqr,
            "auprc_median": auprc_median,
            "auprc_iqr": auprc_iqr,
            "auprc_best": auprc_best,
            "per_run": per_run,
        },
        "hscv_fixed": {
            "n_voted": len(voted_cfds),
            "n_validated": len(validated_hscv),
            "n_groups": hscv_n_groups,
            "f1": float(metrics_hscv['f1']),
            "auprc": float(hscv_auprc),
            "has_consistency": hscv_has_consistency,
        },
        "baselines": {
            "eif_auprc": float(auprc_eif),
            "copod_auprc": float(auprc_copod),
        },
        "fusion": {
            "llm_cfd_best_copod_auprc": float(auprc_fusion_llm_copod),
            "hscv_copod_auprc": float(auprc_fusion_hscv_copod),
        },
        "cfdminer_bl": {
            "f1": float(cfdminer_metrics['f1']),
            "precision": float(cfdminer_metrics['precision']),
            "recall": float(cfdminer_metrics['recall']),
            "n_candidates": len(cfdminer_cfds),
        },
    }

    output_file = os.path.join(RESULTS_DIR, "unified_results.json")
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {output_file}")

    # ================================================================
    # SUMMARY TABLE
    # ================================================================
    print("\n" + "=" * 80)
    print("SUMMARY TABLE: All methods side by side")
    print("=" * 80)
    print(f"\n{'Method':<45} {'F1':>8} {'AUPRC':>10}")
    print("-" * 65)
    print(f"{'Individual LLM-CFD (median)':<45} {f1_median:>8.3f} {auprc_median:>10.4f}")
    print(f"{'Individual LLM-CFD (best)':<45} {'---':>8} {auprc_best:>10.4f}")
    print(f"{'HSCV fixed (cat>=50%, top3_mean)':<45} {metrics_hscv['f1']:>8.3f} {hscv_auprc:>10.4f}")
    print(f"{'EIF baseline':<45} {'---':>8} {auprc_eif:>10.4f}")
    print(f"{'COPOD baseline':<45} {'---':>8} {auprc_copod:>10.4f}")
    print(f"{'LLM-CFD best + COPOD fusion':<45} {'---':>8} {auprc_fusion_llm_copod:>10.4f}")
    print(f"{'HSCV + COPOD fusion':<45} {'---':>8} {auprc_fusion_hscv_copod:>10.4f}")
    print(f"{'CFDMiner-BL (discovery only)':<45} {cfdminer_metrics['f1']:>8.3f} {'---':>10}")
    print("-" * 65)


if __name__ == "__main__":
    main()
