#!/usr/bin/env python3
"""
Self-Consistency Voting v1 — Base module for LLM-CFD self-consistency voting.

Provides shared constants, data loading, CFD validation, anomaly scoring,
and baseline implementations reused by self_consistency_voting_v2.py and
other experiment scripts.

Usage:
    cd experiments
    python3 src/self_consistency_voting.py
"""
import json
import os
import re
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import average_precision_score
from scipy import stats as sp_stats

warnings.filterwarnings('ignore')

# ============================================================
# PATHS & CONSTANTS
# ============================================================
DATA_PATH = "/sessions/69e790d3616697199023cd0a/workspace/mqw7dr6nxdtvfujn/experiments/data/telco_churn.csv"
CACHE_DIR = "/sessions/69e790d3616697199023cd0a/workspace/mqw7dr6nxdtvfujn/experiments/results/supplementary/cache"
RESULTS_PATH = "/sessions/69e790d3616697199023cd0a/workspace/mqw7dr6nxdtvfujn/experiments/results"
FIGURES_PATH = "/sessions/69e790d3616697199023cd0a/workspace/mqw7dr6nxdtvfujn/experiments/results/figures"

SUPPORT_THRESHOLD = 0.01
CONFIDENCE_THRESHOLDS = {"high": 0.90, "medium": 0.85, "low": 0.75}
RANDOM_SEED = 42

TYPE_MAP = {
    'fd': 'categorical', 'enum': 'categorical', 'logic': 'categorical',
    'range': 'numerical', 'consistency': 'structural',
}
CATEGORICAL_TYPES = {'fd', 'enum', 'logic'}

GROUND_TRUTH_CFDS = [
    {"dependent_attribute": "OnlineSecurity", "condition_attributes": ["InternetService"], "type": "fd"},
    {"dependent_attribute": "OnlineBackup", "condition_attributes": ["InternetService"], "type": "fd"},
    {"dependent_attribute": "DeviceProtection", "condition_attributes": ["InternetService"], "type": "fd"},
    {"dependent_attribute": "TechSupport", "condition_attributes": ["InternetService"], "type": "fd"},
    {"dependent_attribute": "StreamingTV", "condition_attributes": ["InternetService"], "type": "fd"},
    {"dependent_attribute": "StreamingMovies", "condition_attributes": ["InternetService"], "type": "fd"},
    {"dependent_attribute": "MultipleLines", "condition_attributes": ["PhoneService"], "type": "fd"},
    {"dependent_attribute": "MonthlyCharges", "condition_attributes": ["Contract"], "type": "range"},
    {"dependent_attribute": "MonthlyCharges", "condition_attributes": [], "type": "range"},
    {"dependent_attribute": "SeniorCitizen", "condition_attributes": [], "type": "enum"},
    {"dependent_attribute": "gender", "condition_attributes": [], "type": "enum"},
    {"dependent_attribute": "Partner", "condition_attributes": [], "type": "enum"},
    {"dependent_attribute": "Dependents", "condition_attributes": [], "type": "enum"},
    {"dependent_attribute": "tenure", "condition_attributes": [], "type": "logic"},
    {"dependent_attribute": "TotalCharges", "condition_attributes": ["tenure", "MonthlyCharges"], "type": "consistency"},
]

os.makedirs(RESULTS_PATH, exist_ok=True)
os.makedirs(FIGURES_PATH, exist_ok=True)


# ============================================================
# CFD KEY & DATA LOADING
# ============================================================
def cfd_key(c):
    """Normalized key: (dep_attr, sorted_cond_attrs, type_category)."""
    raw_type = c.get("type", "fd")
    return (c.get("dependent_attribute", ""),
            tuple(sorted(c.get("condition_attributes", []))),
            TYPE_MAP.get(raw_type, raw_type))


def load_data():
    """Load and preprocess Telco Churn dataset."""
    df = pd.read_csv(DATA_PATH)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
    df = df.drop(columns=['customerID'])
    return df


def inject_anomalies(df, seed=42):
    """Inject synthetic anomalies: dependency destroy 10% + range 5% + logic 3%."""
    np.random.seed(seed)
    df_out = df.copy()
    n = len(df_out)
    labels = np.zeros(n)

    n_dep = int(n * 0.10)
    dep_idx = np.random.choice(n, n_dep, replace=False)
    for idx in dep_idx:
        swap_idx = np.random.choice(n)
        df_out.loc[idx, 'MonthlyCharges'] = df_out.loc[swap_idx, 'MonthlyCharges']
        labels[idx] = 1

    n_range = int(n * 0.05)
    range_idx = np.random.choice(n, n_range, replace=False)
    for idx in range_idx:
        col = np.random.choice(['MonthlyCharges', 'tenure'])
        # In-range distributional outlier (CFD-legal, not 3σ extreme)
        p = np.random.choice([1, 2, 98, 99])
        shifted_val = np.percentile(df[col].astype(float), p)
        shifted_val += np.random.normal(0, abs(shifted_val) * 0.02)
        col_min = df[col].astype(float).min()
        col_max = df[col].astype(float).max()
        df_out.loc[idx, col] = np.clip(shifted_val, col_min, col_max)
        labels[idx] = 1

    n_logic = int(n * 0.03)
    logic_idx = np.random.choice(n, n_logic, replace=False)
    for idx in logic_idx:
        df_out.loc[idx, 'tenure'] = -np.random.randint(1, 20)
        labels[idx] = 1

    return df_out, labels


# ============================================================
# CACHE PARSING
# ============================================================
def _load_cached_response(cache_file):
    """Load and clean a cached LLM response from file."""
    with open(cache_file) as f:
        raw = f.read()
    try:
        decoded = json.loads(raw)
        if isinstance(decoded, str):
            return decoded
        elif isinstance(decoded, dict) and 'response' in decoded:
            return decoded['response']
    except (json.JSONDecodeError, ValueError):
        pass
    raw = raw.strip()
    if raw.startswith('"'):
        raw = raw[1:]
    if raw.endswith('"'):
        raw = raw[:-1]
    raw = raw.replace('\\n', '\n').replace('\\"', '"')
    if raw.startswith('```json'):
        raw = raw[7:]
    elif raw.startswith('```'):
        raw = raw[3:]
    if raw.endswith('```'):
        raw = raw[:-3]
    return raw.strip()


def _parse_cfds_robust(response_text, valid_columns):
    """Parse CFDs from LLM response with truncation fallback."""
    # Try standard JSON parse first
    json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
    else:
        json_match = re.search(r'\[\s*\{.*', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
        else:
            return []

    try:
        cfds = json.loads(json_str)
        return [c for c in cfds if isinstance(c, dict)
                and 'type' in c and 'dependent_attribute' in c
                and c['dependent_attribute'] in valid_columns]
    except (json.JSONDecodeError, ValueError):
        pass

    # Fallback: brace-depth tracking for truncated JSON
    objects = []
    depth = 0
    start = None
    for i, ch in enumerate(json_str):
        if ch == '{':
            if depth == 0:
                start = i
            depth += 1
        elif ch == '}':
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


def parse_cfds_from_cache(cache_dir, valid_columns):
    """Parse CFDs from all cache files in a directory.
    Returns a list of (cache_filename, cfds) tuples.
    """
    results = []
    cache_files = sorted([f for f in os.listdir(cache_dir) if f.endswith('.json')])
    for cf in cache_files:
        response = _load_cached_response(os.path.join(cache_dir, cf))
        cfds = _parse_cfds_robust(response, valid_columns)
        results.append((cf, cfds))
    return results


def load_all_runs(cache_dir, valid_columns):
    """Load all CFD runs from cache directory.
    Returns a list of CFD lists (one per cache file).
    """
    runs = []
    for _, cfds in parse_cfds_from_cache(cache_dir, valid_columns):
        runs.append(cfds)
    return runs


# ============================================================
# CFD EVALUATION & VALIDATION
# ============================================================
def evaluate_condition(df, cfd):
    """Evaluate which rows satisfy a CFD's condition. Returns pd.Series of bool."""
    cond_attrs = cfd.get('condition_attributes', [])
    cond_vals = cfd.get('condition_values', {})
    if not cond_attrs:
        return pd.Series([True] * len(df), index=df.index)
    mask = pd.Series([True] * len(df), index=df.index)
    for attr in cond_attrs:
        if attr in df.columns and attr in cond_vals:
            mask = mask & (df[attr].astype(str) == str(cond_vals[attr]))
    return mask


def evaluate_dependency(df, cfd, condition_mask):
    """Evaluate which rows satisfy a CFD's dependency. Returns pd.Series of bool."""
    dep = cfd.get('dependent_attribute', '')
    cfd_type = cfd.get('type', 'fd')
    ep = cfd.get('expected_pattern', {})
    if isinstance(ep, str):
        ep = {'expression': ep} if cfd_type in ('logic', 'fd') else {}

    if dep not in df.columns:
        return pd.Series([True] * len(df), index=df.index)

    # Handle both pd.Series and np.ndarray for condition_mask
    if isinstance(condition_mask, np.ndarray):
        condition_mask = pd.Series(condition_mask, index=df.index)

    sub = df[condition_mask]
    satisfied = pd.Series([True] * len(sub), index=sub.index)

    if cfd_type == 'range':
        mn = ep.get('min', -float('inf'))
        mx = ep.get('max', float('inf'))
        try:
            vals = pd.to_numeric(sub[dep], errors='coerce')
            satisfied = (vals >= mn) & (vals <= mx)
        except Exception:
            pass
    elif cfd_type in ('enum', 'fd'):
        expected = ep.get('values', [])
        if expected:
            satisfied = sub[dep].astype(str).isin([str(v) for v in expected])
    elif cfd_type == 'consistency':
        if dep == 'TotalCharges' and all(c in df.columns for c in ['tenure', 'MonthlyCharges']):
            expected_vals = sub['tenure'] * sub['MonthlyCharges']
            actual_vals = pd.to_numeric(sub[dep], errors='coerce')
            satisfied = (abs(actual_vals - expected_vals) / (expected_vals.abs() + 1)).clip(0, 1) < 0.2
    elif cfd_type == 'logic':
        expr = ep.get('expression', '')
        if 'tenure>24' in expr and 'Churn' in df.columns:
            satisfied = ~((sub['tenure'] > 24) & (sub['Churn'] == 'Yes'))
        elif 'tenure<6' in expr and 'Churn' in df.columns:
            satisfied = ~((sub['tenure'] < 6) & (sub['Churn'] == 'No'))
        else:
            satisfied = pd.Series([True] * len(sub), index=sub.index)

    full_satisfied = pd.Series([True] * len(df), index=df.index)
    full_satisfied[condition_mask] = satisfied
    return full_satisfied


def validate_cfds(df, candidates):
    """Validate CFD candidates using support and confidence thresholds.
    Returns list of validated CFDs (with support/confidence added).
    """
    n_total = len(df)
    validated = []
    for cfd in candidates:
        condition_mask = evaluate_condition(df, cfd)
        condition_mask = pd.Series(condition_mask.values, index=df.index)
        support = condition_mask.sum() / n_total
        if support < SUPPORT_THRESHOLD:
            continue
        dep_satisfied = evaluate_dependency(df, cfd, condition_mask)
        n_cond = condition_mask.sum()
        n_violations = (~dep_satisfied[condition_mask]).sum()
        confidence = 1.0 - (n_violations / n_cond) if n_cond > 0 else 0.0
        llm_conf = cfd.get('confidence_estimate', 'medium')
        threshold = CONFIDENCE_THRESHOLDS.get(llm_conf, 0.85)
        if confidence >= threshold:
            cfd['support'] = support
            cfd['confidence'] = confidence
            validated.append(cfd)

    # Deduplicate by cfd_key, keeping highest confidence
    seen = {}
    for c in validated:
        key = cfd_key(c)
        if key not in seen or c.get('confidence', 0) > seen[key].get('confidence', 0):
            seen[key] = c
    return list(seen.values())


# ============================================================
# ANOMALY SCORING
# ============================================================
def compute_anomaly_scores(df, validated_cfds):
    """Build violation matrix and compute weighted anomaly scores."""
    if not validated_cfds:
        return np.zeros(len(df))

    n = len(df)
    m = len(validated_cfds)
    V = np.zeros((n, m))

    for j, cfd in enumerate(validated_cfds):
        condition_mask = evaluate_condition(df, cfd)
        dep_attr = cfd.get("dependent_attribute", "")
        if dep_attr not in df.columns:
            continue
        if not condition_mask.any():
            continue

        dep_type = cfd.get("type", "fd")
        expected = cfd.get("expected_pattern", {})

        if dep_type == "range":
            lo = expected.get("min", -np.inf)
            hi = expected.get("max", np.inf)
            values = df[dep_attr].astype(float).values
            rng = hi - lo + 1e-8
            below = np.maximum(lo - values, 0) / rng
            above = np.maximum(values - hi, 0) / rng
            V[:, j] = condition_mask.astype(float).values * np.maximum(below, above)
        elif dep_type == "enum":
            allowed = set(str(v) for v in expected.get("values", []))
            sub_vals = df.loc[condition_mask, dep_attr].astype(str)
            vc = sub_vals.value_counts()
            total = len(sub_vals)
            if total > 0:
                for idx_pos, val in zip(np.where(condition_mask.values)[0], sub_vals.values):
                    freq = vc.get(val, 0) / total
                    V[idx_pos, j] = 1.0 - freq
            violated = ~df[dep_attr].astype(str).isin(allowed).values
            V[:, j] = np.where(violated & condition_mask.values, 1.0, V[:, j])
        elif dep_type == "consistency":
            if dep_attr == "TotalCharges" and all(c in df.columns for c in ['tenure', 'MonthlyCharges']):
                actual = pd.to_numeric(df[dep_attr], errors='coerce').values
                expected_vals = pd.to_numeric(df['tenure'], errors='coerce').values * pd.to_numeric(df['MonthlyCharges'], errors='coerce').values
                deviation = np.abs(actual - expected_vals) / (np.abs(expected_vals) + 1e-8)
                V[:, j] = condition_mask.astype(float).values * np.clip(deviation, 0, 1)
            else:
                dep_satisfied = evaluate_dependency(df, cfd, condition_mask)
                full_satisfied = np.ones(n, dtype=bool)
                full_satisfied[condition_mask.values] = dep_satisfied[condition_mask].values
                V[:, j] = condition_mask.astype(float).values * (~full_satisfied).astype(float)
        elif dep_type == "logic":
            expr = expected.get("expression", "")
            if "tenure" in expr and dep_attr == "tenure":
                vals = pd.to_numeric(df[dep_attr], errors='coerce').values
                sub_vals = pd.to_numeric(df.loc[condition_mask, dep_attr], errors='coerce')
                if len(sub_vals) > 0:
                    for idx_pos, val in zip(np.where(condition_mask.values)[0], sub_vals.values):
                        V[idx_pos, j] = 1.0 if val < 0 else 0.0
            else:
                dep_satisfied = evaluate_dependency(df, cfd, condition_mask)
                full_satisfied = np.ones(n, dtype=bool)
                full_satisfied[condition_mask.values] = dep_satisfied[condition_mask].values
                V[:, j] = condition_mask.astype(float).values * (~full_satisfied).astype(float)
        else:
            # fd: gradient 1 - P(dep=actual | condition)
            sub_vals = df.loc[condition_mask, dep_attr].astype(str)
            vc = sub_vals.value_counts()
            total = len(sub_vals)
            if total > 0:
                for idx_pos, val in zip(np.where(condition_mask.values)[0], sub_vals.values):
                    freq = vc.get(val, 0) / total
                    V[idx_pos, j] = 1.0 - freq

    # Redundancy reduction: group by dependent_attribute, take max per group
    dep_groups = {}
    for j, cfd in enumerate(validated_cfds):
        dep = cfd.get("dependent_attribute", f"unknown_{j}")
        if dep not in dep_groups:
            dep_groups[dep] = []
        dep_groups[dep].append(j)
    n_groups = len(dep_groups)
    V_reduced = np.zeros((n, n_groups))
    for g_idx, (dep, indices) in enumerate(sorted(dep_groups.items())):
        V_reduced[:, g_idx] = V[:, indices].max(axis=1)
    # top3_mean aggregation
    k = min(3, n_groups)
    if n_groups <= k:
        scores = V_reduced.mean(axis=1)
    else:
        topk = np.sort(V_reduced, axis=1)[:, -k:]
        scores = topk.mean(axis=1)

    clip_val = np.percentile(scores, 99)
    scores = np.clip(scores, 0, clip_val)
    if scores.max() > 0:
        scores = scores / scores.max()
    return scores


# ============================================================
# DISCOVERY EVALUATION
# ============================================================
def evaluate_discovery(discovered, ground_truth):
    """Evaluate CFD discovery: Precision, Recall, F1 (type-normalized matching)."""
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


# ============================================================
# BASELINES
# ============================================================
def compute_eif_scores(df_anom):
    """Extended Isolation Forest baseline."""
    df_enc = df_anom.copy()
    for col in df_enc.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df_enc[col] = le.fit_transform(df_enc[col].astype(str))
    iforest = IsolationForest(n_estimators=200, contamination=0.18, random_state=RANDOM_SEED)
    iforest.fit(df_enc)
    return -iforest.score_samples(df_enc)


def copod_scores(df_anom):
    """COPOD: Copula-Based Outlier Detection (from scratch)."""
    df_enc = df_anom.copy()
    for col in df_enc.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df_enc[col] = le.fit_transform(df_enc[col].astype(str))

    X = df_enc.values.astype(float)
    n, d = X.shape

    U_l = np.zeros((n, d))
    U_r = np.zeros((n, d))
    for j in range(d):
        col = X[:, j]
        ranks = sp_stats.rankdata(col, method='average')
        U_l[:, j] = ranks / (n + 1)
        U_r[:, j] = 1.0 - U_l[:, j]

    eps = 1e-10
    U_l = np.clip(U_l, eps, 1 - eps)
    U_r = np.clip(U_r, eps, 1 - eps)

    left_tail = -np.log(U_l)
    right_tail = -np.log(U_r)

    skewness = np.array([sp_stats.skew(X[:, j]) for j in range(d)])
    scores = np.zeros(n)
    for j in range(d):
        if skewness[j] < 0:
            scores += left_tail[:, j]
        else:
            scores += right_tail[:, j]
    return scores


# ============================================================
# MAJORITY VOTING
# ============================================================
def majority_vote(all_runs_cfds, min_votes=None):
    """Aggregate multiple LLM runs using majority voting.

    Categorical types (fd/enum/logic): majority vote (>= ceil(n_runs * 0.5))
    Numerical/structural types (range/consistency): union (>= 1 run)

    Parameters
    ----------
    all_runs_cfds : list of list of dict
        CFD lists from each LLM run.
    min_votes : int or None
        Minimum votes for categorical rules. If None, uses ceil(n_runs * 0.5).

    Returns
    -------
    list of dict
        Voted CFDs (best representative from each surviving key).
    """
    n_runs = len(all_runs_cfds)
    if min_votes is None:
        min_votes = int(np.ceil(n_runs * 0.5))

    key_to_info = {}
    for run_idx, cfds in enumerate(all_runs_cfds):
        seen = set()
        for cfd in cfds:
            raw_type = cfd.get("type", "fd")
            key = (cfd.get("dependent_attribute", ""),
                   tuple(sorted(cfd.get("condition_attributes", []))),
                   raw_type)
            if key in seen:
                continue
            seen.add(key)
            if key not in key_to_info:
                key_to_info[key] = {'cfds': [], 'count': 0, 'orig_type': raw_type}
            key_to_info[key]['cfds'].append(cfd)
            key_to_info[key]['count'] += 1

    voted_cfds = []
    for key, info in key_to_info.items():
        is_cat = info['orig_type'] in CATEGORICAL_TYPES
        if (is_cat and info['count'] >= min_votes) or (not is_cat and info['count'] >= 1):
            best = max(info['cfds'], key=lambda c:
                       {'high': 3, 'medium': 2, 'low': 1}.get(
                           c.get('confidence_estimate', 'low'), 0))
            voted_cfds.append(best)
    return voted_cfds


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 70)
    print("Self-Consistency Voting Experiment")
    print("=" * 70)

    df = load_data()
    df_clean = df.copy()
    df_anom, labels = inject_anomalies(df, seed=RANDOM_SEED)
    valid_columns = set(df.columns)
    print(f"Data: {df.shape}, Anomalies: {int(labels.sum())}")

    # Load all cached runs
    all_runs = load_all_runs(CACHE_DIR, valid_columns)
    print(f"Loaded {len(all_runs)} cached runs")

    if not all_runs:
        print("No cache files found. Run experiment.py first.")
        return

    # Individual run metrics
    print("\n--- Individual Runs ---")
    per_run_auprc = []
    for i, cfds in enumerate(all_runs):
        validated = validate_cfds(df_clean, [c.copy() for c in cfds])
        metrics = evaluate_discovery(validated, GROUND_TRUTH_CFDS)
        scores = compute_anomaly_scores(df_anom, validated)
        auprc = average_precision_score(labels, scores)
        per_run_auprc.append(auprc)
        print(f"  Run {i+1}: n_cfd={len(cfds)} n_val={len(validated)} "
              f"F1={metrics['f1']:.3f} AUPRC={auprc:.3f}")

    # Majority vote
    print("\n--- Majority Vote (HSCV) ---")
    voted = majority_vote(all_runs)
    validated_voted = validate_cfds(df_clean, [c.copy() for c in voted])
    metrics_voted = evaluate_discovery(validated_voted, GROUND_TRUTH_CFDS)
    scores_voted = compute_anomaly_scores(df_anom, validated_voted)
    auprc_voted = average_precision_score(labels, scores_voted)
    print(f"  Voted: n_voted={len(voted)} n_val={len(validated_voted)} "
          f"F1={metrics_voted['f1']:.3f} AUPRC={auprc_voted:.3f}")

    # Baselines
    print("\n--- Baselines ---")
    eif_scores = compute_eif_scores(df_anom)
    copod_s = copod_scores(df_anom)
    print(f"  EIF AUPRC:   {average_precision_score(labels, eif_scores):.4f}")
    print(f"  COPOD AUPRC: {average_precision_score(labels, copod_s):.4f}")

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"  Individual AUPRC: median={np.median(per_run_auprc):.3f} "
          f"IQR={np.percentile(per_run_auprc,75)-np.percentile(per_run_auprc,25):.3f}")
    print(f"  HSCV AUPRC:       {auprc_voted:.3f}")


if __name__ == "__main__":
    main()
