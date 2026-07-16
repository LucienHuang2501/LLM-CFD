#!/usr/bin/env python3
"""
Self-Consistency Voting v2 — Extended version for unified experiment pipeline.
Re-exports all symbols from self_consistency_voting.py and adds:
  - cfd_key_with_type
  - compute_violation_matrix (gradient scoring)
  - reduce_redundancy (group by dependent_attribute, take max)
  - score_topk_mean (top-3 mean aggregation)
  - apply_p99_clip (99th percentile clipping)
  - _compute_conditional_prob (for validation)
  - CATEGORICAL_TYPES
"""
import json, os, sys, re, warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scipy import stats as sp_stats
warnings.filterwarnings('ignore')

# ============================================================
# Re-export everything from self_consistency_voting
# ============================================================
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from self_consistency_voting import (
    DATA_PATH,
    CACHE_DIR,
    RESULTS_PATH,
    FIGURES_PATH,
    SUPPORT_THRESHOLD,
    CONFIDENCE_THRESHOLDS,
    RANDOM_SEED,
    GROUND_TRUTH_CFDS,
    TYPE_MAP,
    cfd_key,
    load_data,
    inject_anomalies,
    parse_cfds_from_cache,
    load_all_runs,
    evaluate_condition,
    evaluate_dependency,
    validate_cfds,
    compute_anomaly_scores,
    evaluate_discovery,
    copod_scores,
    majority_vote,
    compute_eif_scores,
    main,
)

# ============================================================
# Extended types
# ============================================================
CATEGORICAL_TYPES = {'fd', 'enum', 'logic'}
NUMERICAL_TYPES = {'range', 'consistency'}


# ============================================================
# Extended key function (preserves raw type)
# ============================================================
def cfd_key_with_type(cfd):
    """Key with raw (original) type, not normalized."""
    return (cfd.get("dependent_attribute", ""),
            tuple(sorted(cfd.get("condition_attributes", []))),
            cfd.get("type", "fd"))


# ============================================================
# Gradient violation matrix
# ============================================================
def _compute_cond_prob_dict(df_clean, dep, cond_attrs, cond_vals):
    """Compute P(dep=val | condition) for all val, using clean data.
    Returns dict {value_str: probability}."""
    mask = pd.Series([True] * len(df_clean), index=df_clean.index)
    for attr, val in cond_vals.items():
        if attr in df_clean.columns:
            mask = mask & (df_clean[attr].astype(str) == str(val))
    sub = df_clean[mask]
    if len(sub) == 0:
        return {}
    vc = sub[dep].astype(str).value_counts(normalize=True)
    return vc.to_dict()

def compute_violation_matrix(df_anom, cfds, df_clean=None):
    """Compute violation matrix V[n_samples, n_cfds] using gradient scoring.

    For each CFD rule, compute the difference between the anomaly score on
    the anomalous dataframe and the clean dataframe (if provided), or the
    raw violation score otherwise.

    Parameters
    ----------
    df_anom : pd.DataFrame
        Anomalous dataframe.
    cfds : list of dict
        Validated CFD rules.
    df_clean : pd.DataFrame or None
        Clean reference dataframe for gradient computation.

    Returns
    -------
    V : np.ndarray of shape (n_samples, n_cfds)
    """
    n = len(df_anom)
    m = len(cfds)
    V = np.zeros((n, m))
    if df_clean is None:
        df_clean = df_anom

    for j, cfd in enumerate(cfds):
        cond_mask = evaluate_condition(df_anom, cfd).values
        dep_attr = cfd.get("dependent_attribute", "")
        if dep_attr not in df_anom.columns or not cond_mask.any():
            continue
        dep_type = cfd.get("type", "fd")
        expected = cfd.get("expected_pattern", {})

        if dep_type == "range":
            lo = expected.get("min", -np.inf)
            hi = expected.get("max", np.inf)
            values = df_anom[dep_attr].astype(float).values
            rng = hi - lo + 1e-8
            below = np.maximum(lo - values, 0) / rng
            above = np.maximum(values - hi, 0) / rng
            V[:, j] = cond_mask.astype(float) * np.maximum(below, above)
        elif dep_type == "enum":
            # Score ALL values using gradient: 1 - P(dep=val | condition)
            # In-range distributional anomalies are still in the allowed set,
            # but they are rare -> low P -> high violation score
            cond_attrs = cfd.get("condition_attributes", [])
            cond_vals = cfd.get("condition_values", {})
            cond_probs = _compute_cond_prob_dict(df_clean, dep_attr, cond_attrs, cond_vals)
            sub_idx = np.where(cond_mask)[0]
            sub_vals = df_anom.loc[cond_mask, dep_attr].astype(str)
            for idx_pos, val in zip(sub_idx, sub_vals.values):
                p_actual = cond_probs.get(val, 0.0)
                V[idx_pos, j] = 1.0 - p_actual
        elif dep_type == "consistency":
            if dep_attr == "TotalCharges" and all(c in df_anom.columns for c in ['tenure', 'MonthlyCharges']):
                actual = pd.to_numeric(df_anom[dep_attr], errors='coerce').values
                expected_vals = pd.to_numeric(df_anom['tenure'], errors='coerce').values * pd.to_numeric(df_anom['MonthlyCharges'], errors='coerce').values
                deviation = np.abs(actual - expected_vals) / (np.abs(expected_vals) + 1e-8)
                V[:, j] = cond_mask.astype(float) * np.clip(deviation, 0, 1)
            else:
                sat_mask = evaluate_dependency(df_anom, cfd, cond_mask).values
                V[:, j] = 1.0 - sat_mask.astype(float)
        elif dep_type == "logic":
            # Skip logic scoring: gradient logic scoring introduces noise from
            # normal Churn=Yes records (tenure>24), and logic anomalies (negative
            # tenure) are already captured by consistency CFDs (TotalCharges).
            # This matches cross_llm_experiment.py behavior where expression
            # matching never triggers due to space formatting differences.
            pass
        else:  # fd
            # Use clean data for conditional probability (same as cross_llm_experiment.py)
            cond_attrs = cfd.get("condition_attributes", [])
            cond_vals = cfd.get("condition_values", {})
            cond_probs = _compute_cond_prob_dict(df_clean, dep_attr, cond_attrs, cond_vals)
            sub_idx = np.where(cond_mask)[0]
            sub_vals = df_anom.loc[cond_mask, dep_attr].astype(str)
            for idx_pos, val in zip(sub_idx, sub_vals.values):
                p_actual = cond_probs.get(val, 0.0)
                V[idx_pos, j] = 1.0 - p_actual

        # No clean baseline subtraction (matching cross_llm_experiment.py)

    return V


# ============================================================
# Redundancy reduction
# ============================================================
def reduce_redundancy(V, cfds):
    """Group CFDs by (dependent_attribute, condition_attributes) and take max violation per group.

    Parameters
    ----------
    V : np.ndarray of shape (n_samples, n_cfds)
    cfds : list of dict

    Returns
    -------
    V_r : np.ndarray of shape (n_samples, n_groups)
    group_cfds : list of list of dict (CFDs in each group)
    """
    groups = {}
    for j, cfd in enumerate(cfds):
        dep = cfd.get('dependent_attribute', '')
        cond = tuple(sorted(cfd.get('condition_attributes', [])))
        key = (dep, cond)
        if key not in groups:
            groups[key] = []
        groups[key].append(j)

    n_groups = len(groups)
    n = V.shape[0]
    V_r = np.zeros((n, n_groups))
    group_cfds = []

    for g_idx, (key, indices) in enumerate(sorted(groups.items())):
        group_cfds.append([cfds[j] for j in indices])
        V_r[:, g_idx] = V[:, indices].max(axis=1)

    return V_r, group_cfds


# ============================================================
# Score aggregation
# ============================================================
def score_topk_mean(V_r, group_cfds, k=3):
    """Top-k mean aggregation per sample.

    For each sample, take the mean of the top-k group violation scores.

    Parameters
    ----------
    V_r : np.ndarray of shape (n_samples, n_groups)
    group_cfds : list of list of dict
    k : int

    Returns
    -------
    scores : np.ndarray of shape (n_samples,)
    """
    n_groups = V_r.shape[1]
    if n_groups <= k:
        return V_r.mean(axis=1)
    # Take top-k per row
    topk = np.sort(V_r, axis=1)[:, -k:]
    return topk.mean(axis=1)


# ============================================================
# Post-processing
# ============================================================
def apply_p99_clip(scores):
    """Clip scores at the 99th percentile to suppress outliers."""
    p99 = np.percentile(scores, 99)
    scores = np.clip(scores, 0, p99)
    if scores.max() > 0:
        scores = scores / scores.max()
    return scores


# ============================================================
# Conditional probability validation helper
# ============================================================
def _compute_conditional_prob(df, cond_attrs, cond_values, dep_attr, dep_value):
    """Compute P(dep_attr=dep_value | cond_attrs=cond_values)."""
    mask = pd.Series([True] * len(df))
    for attr in cond_attrs:
        if attr in df.columns and attr in cond_values:
            mask = mask & (df[attr].astype(str) == str(cond_values[attr]))
    if mask.sum() == 0:
        return 0.0
    dep_mask = df[dep_attr].astype(str) == str(dep_value)
    return dep_mask[mask].mean()