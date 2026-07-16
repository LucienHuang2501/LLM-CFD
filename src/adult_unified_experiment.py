#!/usr/bin/env python3
"""
Adult Unified Experiment: All methods under the SAME pipeline as Telco.
======================================================================
Fixes audit issues:
  - Critical 2.2: Adult AUPRC=0.231 untraceable → re-run with unified scoring
  - Critical 2.3: Adult CFDMiner-BL mismatch → re-run with 4×4 sweep
  - Major 1.1: Adult dataset size → document subsampling (32,561→10,000, seed=42)

Key changes from adult_hscv_experiment.py:
  1. Per-run scoring uses score_topk_mean(k=3) + apply_p99_clip (was score_mean)
  2. CFDMiner-BL runs 4×4 sweep (was single config)
  3. Results saved to adult_unified_results.json
"""
import json, os, sys, re, hashlib, warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, precision_recall_curve
from scipy import stats as sp_stats
# combinations removed: single-attribute only (consistent with Telco)
warnings.filterwarnings('ignore')

# ============================================================
# CONFIG
# ============================================================
BASE = "/sessions/69e790d3616697199023cd0a/workspace/mqw7dr6nxdtvfujn"
DATA_PATH = os.path.join(BASE, "experiments/data/census+income/adult.data")
CACHE_DIR = os.path.join(BASE, "experiments/results/p1_revision/cache")
RESULTS_PATH = os.path.join(BASE, "experiments/results/unified")
FIGURES_PATH = os.path.join(BASE, "experiments/results/figures")
os.makedirs(RESULTS_PATH, exist_ok=True)
os.makedirs(FIGURES_PATH, exist_ok=True)

RANDOM_SEED = 42
N_RUNS = 10
SUPPORT_THRESHOLD = 0.01
CONFIDENCE_THRESHOLDS = {"high": 0.90, "medium": 0.85, "low": 0.75}

TYPE_MAP = {'fd': 'categorical', 'enum': 'categorical', 'logic': 'categorical',
            'range': 'numerical', 'consistency': 'structural'}
CATEGORICAL_TYPES = {'fd', 'enum', 'logic'}

# Adult ground truth (12 rules — same as adult_hscv_experiment.py)
ADULT_GROUND_TRUTH = [
    {"dependent_attribute": "education_num", "condition_attributes": ["education"], "type": "fd"},
    {"dependent_attribute": "hours_per_week", "condition_attributes": ["occupation"], "type": "range"},
    {"dependent_attribute": "hours_per_week", "condition_attributes": ["sex"], "type": "range"},
    {"dependent_attribute": "income", "condition_attributes": ["education"], "type": "logic"},
    {"dependent_attribute": "income", "condition_attributes": ["hours_per_week"], "type": "logic"},
    {"dependent_attribute": "marital_status", "condition_attributes": ["age"], "type": "logic"},
    {"dependent_attribute": "relationship", "condition_attributes": ["marital_status"], "type": "fd"},
    {"dependent_attribute": "relationship", "condition_attributes": ["sex"], "type": "fd"},
    {"dependent_attribute": "occupation", "condition_attributes": ["education"], "type": "fd"},
    {"dependent_attribute": "workclass", "condition_attributes": [], "type": "enum"},
    {"dependent_attribute": "race", "condition_attributes": [], "type": "enum"},
    {"dependent_attribute": "native_country", "condition_attributes": [], "type": "enum"},
]

# ============================================================
# UTILITIES
# ============================================================
def cfd_key(c):
    raw_type = c.get("type", "fd")
    return (c.get("dependent_attribute", ""),
            tuple(sorted(c.get("condition_attributes", []))),
            TYPE_MAP.get(raw_type, raw_type))

def cfd_key_with_type(c):
    raw_type = c.get("type", "fd")
    return (c.get("dependent_attribute", ""),
            tuple(sorted(c.get("condition_attributes", []))),
            raw_type)

def evaluate_discovery(discovered, ground_truth):
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

def rank_normalize(scores):
    n = len(scores)
    ranks = sp_stats.rankdata(scores, method='average')
    return ranks / n

def apply_p99_clip(scores):
    p99 = np.percentile(scores, 99)
    if p99 > 0:
        scores = np.clip(scores / p99, 0, 1)
    return scores

def normalize_scores(scores):
    mn, mx = scores.min(), scores.max()
    if mx - mn < 1e-8:
        return np.zeros_like(scores)
    return (scores - mn) / (mx - mn + 1e-8)

# ============================================================
# DATA LOADING (with subsampling documentation)
# ============================================================
def load_adult_data():
    """Load UCI Adult dataset.
    
    Original: 32,561 rows × 15 columns (UCI Machine Learning Repository)
    Subsampled: 10,000 rows (random_state=42) for computational efficiency.
    License: CC BY 4.0 (https://doi.org/10.24432/C5GP7S)
    """
    try:
        df = pd.read_csv(DATA_PATH, skipinitialspace=True, na_values='?')
        if 'age' not in df.columns and 'education' not in df.columns:
            col_names = ['age', 'workclass', 'fnlwgt', 'education', 'education_num',
                         'marital_status', 'occupation', 'relationship', 'race', 'sex',
                         'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income']
            df = pd.read_csv(DATA_PATH, header=None, names=col_names, skipinitialspace=True, na_values='?')
    except:
        col_names = ['age', 'workclass', 'fnlwgt', 'education', 'education_num',
                     'marital_status', 'occupation', 'relationship', 'race', 'sex',
                     'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income']
        df = pd.read_csv(DATA_PATH, header=None, names=col_names, skipinitialspace=True, na_values='?')
    
    df = df.dropna()
    original_rows = len(df)
    df = df.sample(10000, random_state=RANDOM_SEED)
    print(f"Adult data: {df.shape} (subsampled from {original_rows} rows, seed={RANDOM_SEED})")
    return df

def inject_anomalies_adult(df, seed=42):
    """Inject semantic anomalies into Adult dataset (same as adult_hscv_experiment.py)."""
    rng = np.random.RandomState(seed)
    df_out = df.copy().reset_index(drop=True)
    n = len(df_out)
    labels = np.zeros(n, dtype=int)
    
    # 1. Dependency destruction: break education→education_num (10%)
    n_dep = int(n * 0.10)
    dep_idx = rng.choice(n, n_dep, replace=False)
    for idx in dep_idx:
        df_out.loc[idx, 'education_num'] = rng.randint(1, 16)
    labels[dep_idx] = 1
    
    # 2. In-range distributional outlier (5%, CFD-legal)
    n_range = int(n * 0.05)
    avail = list(set(range(n)) - set(dep_idx))
    range_idx = rng.choice(avail, n_range, replace=False)
    for idx in range_idx:
        col = rng.choice(['age', 'hours_per_week', 'capital_gain'])
        # Percentile-based shift within valid range (not 3σ extreme)
        p = rng.choice([1, 2, 98, 99])
        shifted_val = np.percentile(df[col].astype(float), p)
        shifted_val += rng.normal(0, abs(shifted_val) * 0.02)
        col_min = df[col].astype(float).min()
        col_max = df[col].astype(float).max()
        df_out.loc[idx, col] = np.clip(shifted_val, col_min, col_max)
    labels[range_idx] = 1
    
    # 3. Logic contradiction (3%)
    n_logic = int(n * 0.03)
    avail = list(set(range(n)) - set(dep_idx) - set(range_idx))
    logic_idx = rng.choice(avail, n_logic, replace=False)
    for idx in logic_idx:
        choice = rng.choice(['age', 'capital_gain'])
        if choice == 'age':
            df_out.loc[idx, 'age'] = rng.randint(-10, 10)
        else:
            df_out.loc[idx, 'capital_gain'] = rng.randint(-1000, 0)
    labels[logic_idx] = 1
    
    return df_out, labels

# ============================================================
# CFD LOADING & PARSING (from cache)
# ============================================================
def parse_cfds_from_response(response_text, valid_columns):
    """Parse CFDs from LLM response with robust parsing."""
    # Try JSON code block first
    json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
    else:
        json_match = re.search(r'\[\s*\{.*\}\s*\]', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
        else:
            return []
    
    try:
        cfds = json.loads(json_str)
        if isinstance(cfds, list):
            return [c for c in cfds if 'type' in c and 'dependent_attribute' in c
                    and c['dependent_attribute'] in valid_columns]
    except:
        pass
    
    # Fallback: brace-depth tracking
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
        except:
            pass
    return cfds

def load_all_runs(valid_columns):
    """Load all cached LLM responses."""
    cache_files = sorted([f for f in os.listdir(CACHE_DIR) if f.endswith('.json') and f.startswith('adult_')])
    print(f"Found {len(cache_files)} cache files")
    
    all_runs = []
    for cf in cache_files:
        with open(os.path.join(CACHE_DIR, cf)) as f:
            response = f.read()
        cfds = parse_cfds_from_response(response, valid_columns)
        all_runs.append({'file': cf, 'cfds': cfds})
        print(f"  {cf}: {len(cfds)} CFDs")
    
    return all_runs

# ============================================================
# STATISTICAL VALIDATION (same as adult_hscv_experiment.py)
# ============================================================
def evaluate_condition(df, cfd):
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
    dep = cfd.get('dependent_attribute', '')
    cfd_type = cfd.get('type', 'fd')
    ep = cfd.get('expected_pattern', {})
    if isinstance(ep, str):
        ep = {'expression': ep} if cfd_type in ('logic', 'fd') else {}
    
    if dep not in df.columns:
        return pd.Series([True] * len(df), index=df.index)
    
    sub = df[condition_mask]
    satisfied = pd.Series([True] * len(sub), index=sub.index)
    
    if cfd_type == 'range':
        mn = ep.get('min', -float('inf'))
        mx = ep.get('max', float('inf'))
        try:
            vals = pd.to_numeric(sub[dep], errors='coerce')
            satisfied = (vals >= mn) & (vals <= mx)
        except:
            pass
    elif cfd_type in ('enum', 'fd'):
        expected = ep.get('values', [])
        if expected:
            satisfied = sub[dep].astype(str).isin([str(v) for v in expected])
    elif cfd_type == 'consistency':
        if dep == 'education_num' and 'education' in df.columns:
            edu_map = df.groupby('education')['education_num'].agg(lambda x: x.mode().iloc[0] if len(x) > 0 else 0)
            expected = sub['education'].map(edu_map)
            satisfied = (sub[dep].astype(str) == expected.astype(str))
    elif cfd_type == 'logic':
        expr = ep.get('expression', '')
        # Logic rules are validated by confidence only
        pass
    
    full_satisfied = pd.Series([True] * len(df), index=df.index)
    full_satisfied[condition_mask] = satisfied
    return full_satisfied

def validate_cfds(df, candidates):
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
    
    seen = {}
    for c in validated:
        key = cfd_key(c)
        if key not in seen or c.get('confidence', 0) > seen[key].get('confidence', 0):
            seen[key] = c
    return list(seen.values())

# ============================================================
# GRADIENT VIOLATION SCORING (same as adult_hscv_experiment.py)
# ============================================================
def _compute_conditional_prob(df_clean, dep, cond_attrs, cond_vals):
    mask = pd.Series([True] * len(df_clean), index=df_clean.index)
    for attr, val in cond_vals.items():
        if attr in df_clean.columns:
            mask = mask & (df_clean[attr].astype(str) == str(val))
    sub = df_clean[mask]
    if len(sub) == 0:
        return {}
    vc = sub[dep].astype(str).value_counts(normalize=True)
    return vc.to_dict()

def compute_violation_matrix(df, validated_cfds, df_clean=None):
    n = len(df)
    m = len(validated_cfds)
    if m == 0:
        return np.zeros((n, 0))
    if df_clean is None:
        df_clean = df
    
    V = np.zeros((n, m))
    for j, cfd in enumerate(validated_cfds):
        cond_mask = evaluate_condition(df, cfd)
        dep = cfd.get('dependent_attribute', '')
        cfd_type = cfd.get('type', 'fd')
        ep = cfd.get('expected_pattern', {})
        if isinstance(ep, str):
            ep = {'expression': ep} if cfd_type in ('logic', 'fd') else {}
        cond_attrs = cfd.get('condition_attributes', [])
        cond_vals = cfd.get('condition_values', {})
        
        if dep not in df.columns:
            continue
        sub_idx = cond_mask[cond_mask].index
        
        if cfd_type == 'range':
            mn = ep.get('min', df[dep].min())
            mx = ep.get('max', df[dep].max())
            try:
                vals = pd.to_numeric(df.loc[sub_idx, dep], errors='coerce')
                violations = np.maximum(0, mn - vals) + np.maximum(0, vals - mx)
                rng = mx - mn + 1e-8
                V[sub_idx, j] = (violations / rng).clip(0, 1).values
            except:
                pass
        elif cfd_type in ('enum', 'fd'):
            # Score ALL values using gradient: 1 - P(dep=val | condition)
            cond_probs = _compute_conditional_prob(df_clean, dep, cond_attrs, cond_vals)
            for idx in sub_idx:
                actual = str(df.loc[idx, dep])
                p_actual = cond_probs.get(actual, 0.0)
                V[idx, j] = 1.0 - p_actual
        elif cfd_type == 'consistency':
            if dep == 'education_num' and 'education' in df.columns:
                edu_map = df_clean.groupby('education')['education_num'].agg(
                    lambda x: x.mode().iloc[0] if len(x) > 0 else 0)
                expected_vals = df.loc[sub_idx, 'education'].map(edu_map)
                actual_vals = pd.to_numeric(df.loc[sub_idx, dep], errors='coerce')
                expected_num = pd.to_numeric(expected_vals, errors='coerce')
                diff = (actual_vals - expected_num).abs()
                V[sub_idx, j] = (diff / 15.0).clip(0, 1).values
        elif cfd_type == 'logic':
            expr = ep.get('expression', '')
            if 'income' in expr and 'education' in expr:
                for idx in sub_idx:
                    edu = df.loc[idx, 'education']
                    clean_mask = df_clean['education'] == edu
                    if clean_mask.sum() > 10:
                        p_high = (df_clean[clean_mask]['income'].str.strip() == '>50K').mean()
                    else:
                        p_high = 0.3
                    actual_income = str(df.loc[idx, 'income']).strip()
                    if '>50K' in expr and actual_income == '<=50K':
                        V[idx, j] = p_high
                    elif '<=50K' in expr and actual_income == '>50K':
                        V[idx, j] = 1.0 - p_high
            elif 'hours' in expr and 'income' in expr:
                for idx in sub_idx:
                    hpw = df.loc[idx, 'hours_per_week']
                    if hpw > 50:
                        clean_mask = df_clean['hours_per_week'] > 50
                        if clean_mask.sum() > 10:
                            p_high = (df_clean[clean_mask]['income'].str.strip() == '>50K').mean()
                        else:
                            p_high = 0.5
                        actual_income = str(df.loc[idx, 'income']).strip()
                        if actual_income == '<=50K':
                            V[idx, j] = p_high
                    elif hpw < 20:
                        clean_mask = df_clean['hours_per_week'] < 20
                        if clean_mask.sum() > 10:
                            p_low = (df_clean[clean_mask]['income'].str.strip() == '<=50K').mean()
                        else:
                            p_low = 0.5
                        actual_income = str(df.loc[idx, 'income']).strip()
                        if actual_income == '>50K':
                            V[idx, j] = p_low
    return V

def reduce_redundancy(V, validated_cfds):
    if V.shape[1] == 0:
        return V, validated_cfds
    groups = {}
    for j, cfd in enumerate(validated_cfds):
        dep = cfd.get('dependent_attribute', f'unknown_{j}')
        cond = tuple(sorted(cfd.get('condition_attributes', [])))
        key = (dep, cond)
        if key not in groups:
            groups[key] = {'cols': [], 'cfds': []}
        groups[key]['cols'].append(j)
        groups[key]['cfds'].append(cfd)
    
    n = V.shape[0]
    n_groups = len(groups)
    V_reduced = np.zeros((n, n_groups))
    group_cfds = []
    for i, (key, info) in enumerate(groups.items()):
        cols = info['cols']
        V_reduced[:, i] = V[:, cols].max(axis=1)
        best_cfd = max(info['cfds'], key=lambda c: c.get('confidence', 0.5))
        best_cfd['n_merged_rules'] = len(cols)
        group_cfds.append(best_cfd)
    return V_reduced, group_cfds

# ============================================================
# SCORING (UNIFIED: top3_mean + p99_clip)
# ============================================================
def score_topk_mean(V, cfds, k=3):
    if V.shape[1] == 0:
        return np.zeros(V.shape[0])
    k = min(k, V.shape[1])
    if k == 0:
        return np.zeros(V.shape[0])
    topk = np.sort(V, axis=1)[:, -k:]
    return topk.mean(axis=1)

def score_mean(V, cfds):
    if V.shape[1] == 0:
        return np.zeros(V.shape[0])
    confidences = np.array([c.get('confidence', 0.9) for c in cfds])
    return V @ confidences / confidences.sum()

def score_max(V, cfds):
    if V.shape[1] == 0:
        return np.zeros(V.shape[0])
    return V.max(axis=1)

# ============================================================
# HSCV
# ============================================================
def hybrid_vote(all_runs, categorical_threshold=0.5):
    n_runs = len(all_runs)
    cat_min_votes = int(np.ceil(n_runs * categorical_threshold))
    key_to_info = {}
    
    for run_idx, run in enumerate(all_runs):
        seen_in_run = set()
        for cfd in run['cfds']:
            key = cfd_key_with_type(cfd)
            if key in seen_in_run:
                continue
            seen_in_run.add(key)
            if key not in key_to_info:
                key_to_info[key] = {'cfds': [], 'count': 0, 'runs': [], 'original_type': cfd.get('type', 'fd')}
            key_to_info[key]['cfds'].append(cfd)
            key_to_info[key]['count'] += 1
            key_to_info[key]['runs'].append(run_idx)
    
    voted_cfds = []
    for key, info in key_to_info.items():
        orig_type = info['original_type']
        is_categorical = orig_type in CATEGORICAL_TYPES
        if is_categorical:
            if info['count'] >= cat_min_votes:
                best = max(info['cfds'], key=lambda c: {'high': 3, 'medium': 2, 'low': 1}.get(c.get('confidence_estimate', 'low'), 0))
                best['vote_count'] = info['count']
                best['vote_rate'] = info['count'] / n_runs
                best['selection_method'] = 'categorical_vote'
                voted_cfds.append(best)
        else:
            if info['count'] >= 1:
                best = max(info['cfds'], key=lambda c: {'high': 3, 'medium': 2, 'low': 1}.get(c.get('confidence_estimate', 'low'), 0))
                best['vote_count'] = info['count']
                best['vote_rate'] = info['count'] / n_runs
                best['selection_method'] = 'numerical_union'
                voted_cfds.append(best)
    return voted_cfds

# ============================================================
# BASELINES
# ============================================================
def compute_eif_scores(df_anom):
    df_enc = df_anom.copy()
    for col in df_enc.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df_enc[col] = le.fit_transform(df_enc[col].astype(str))
    iforest = IsolationForest(n_estimators=200, contamination=0.18, random_state=RANDOM_SEED)
    iforest.fit(df_enc)
    return -iforest.score_samples(df_enc)

def compute_copod_scores(df_anom):
    df_enc = df_anom.copy()
    for col in df_enc.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df_enc[col] = le.fit_transform(df_enc[col].astype(str))
    X = df_enc.values.astype(float)
    n, d = X.shape
    U_l = np.zeros((n, d))
    U_r = np.zeros((n, d))
    for j in range(d):
        ranks = sp_stats.rankdata(X[:, j], method='average')
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
# CFDMiner-BL SWEEP (4×4 grid, same as Telco)
# ============================================================
def cfdminer_baseline_adult(df, min_support=0.01, min_confidence=0.90):
    """CFDMiner-style baseline for Adult dataset."""
    candidates = []
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    df_enc = df.copy()
    for col in categorical_cols:
        le = LabelEncoder()
        df_enc[col] = le.fit_transform(df[col].astype(str))
    
    # Single attribute conditions only (consistent with Telco CFDMiner-BL)
    condition_patterns = []
    for col in categorical_cols:
        for val in df[col].unique():
            mask = df[col] == val
            support = mask.sum() / len(df)
            if support >= min_support:
                condition_patterns.append({'attrs': [col], 'mask': mask, 'support': support})
    
    # FD check
    for cp in condition_patterns:
        sub = df_enc[cp['mask']]
        if len(sub) < 10:
            continue
        remaining = [c for c in df_enc.columns if c not in cp['attrs']]
        for dep in remaining:
            groups = sub.groupby(cp['attrs'])[dep]
            nunique_per_group = groups.nunique()
            if (nunique_per_group == 1).all() and len(groups) >= 2:
                confidence = 1.0
            else:
                total = len(sub)
                mode_vals = sub.groupby(cp['attrs'])[dep].agg(lambda x: x.value_counts().iloc[0] if len(x) > 0 else 0)
                confidence = mode_vals.sum() / total
            
            if confidence >= min_confidence:
                candidates.append({
                    'type': 'fd',
                    'condition_attributes': cp['attrs'],
                    'dependent_attribute': dep,
                    'confidence': confidence,
                    'support': cp['support'],
                })
    
    # Range constraints
    for col in numerical_cols:
        candidates.append({
            'type': 'range', 'condition_attributes': [], 'dependent_attribute': col,
            'confidence': 1.0, 'support': 1.0,
        })
    
    # Conditional range
    for cat_col in categorical_cols:
        for num_col in numerical_cols:
            for val in df[cat_col].unique():
                mask = df[cat_col] == val
                if mask.sum() >= 30:
                    candidates.append({
                        'type': 'range', 'condition_attributes': [cat_col],
                        'dependent_attribute': num_col,
                        'confidence': 1.0, 'support': mask.sum() / len(df),
                    })
    
    # Enum constraints
    for col in categorical_cols:
        candidates.append({
            'type': 'enum', 'condition_attributes': [], 'dependent_attribute': col,
            'confidence': 1.0, 'support': 1.0,
        })
    
    # Deduplicate
    seen = {}
    for c in candidates:
        key = (c['dependent_attribute'], tuple(sorted(c['condition_attributes'])), c['type'])
        if key not in seen or c['confidence'] > seen[key]['confidence']:
            seen[key] = c
    
    return list(seen.values())

def cfdminer_sweep_adult(df):
    """Run 4×4 sweep: support × confidence."""
    support_list = [0.005, 0.01, 0.02, 0.05]
    confidence_list = [0.80, 0.85, 0.90, 0.95]
    
    results = []
    for min_sup in support_list:
        for min_conf in confidence_list:
            candidates = cfdminer_baseline_adult(df, min_support=min_sup, min_confidence=min_conf)
            metrics = evaluate_discovery(candidates, ADULT_GROUND_TRUTH)
            result = {
                'min_support': min_sup,
                'min_confidence': min_conf,
                'n_candidates': len(candidates),
                **metrics
            }
            results.append(result)
            print(f"  sup={min_sup:.3f} conf={min_conf:.2f} → n={len(candidates):4d} P={metrics['precision']:.3f} R={metrics['recall']:.3f} F1={metrics['f1']:.3f}")
    
    best = max(results, key=lambda x: x['f1'])
    return results, best

# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 80)
    print("ADULT UNIFIED EXPERIMENT: Same pipeline as Telco")
    print("  Scoring: gradient + redundancy reduction + top3_mean + p99_clip")
    print("  CFDMiner-BL: 4×4 hyperparameter sweep")
    print("=" * 80)
    
    # Load data
    df = load_adult_data()
    df_clean = df.copy()
    df_anom, labels = inject_anomalies_adult(df, seed=RANDOM_SEED)
    valid_columns = set(df.columns)
    print(f"Anomalies: {int(labels.sum())} ({labels.mean()*100:.1f}%)")
    
    # Load cached LLM responses
    all_runs = load_all_runs(valid_columns)
    n_runs = len(all_runs)
    
    # Baselines
    eif_scores = compute_eif_scores(df_anom)
    copod_scores = compute_copod_scores(df_anom)
    auprc_eif = average_precision_score(labels, eif_scores)
    auprc_copod = average_precision_score(labels, copod_scores)
    print(f"\nBaselines: EIF AUPRC={auprc_eif:.4f}, COPOD AUPRC={auprc_copod:.4f}")
    
    # ================================================================
    # (a) Individual LLM-CFD runs — UNIFIED scoring (top3_mean + p99_clip)
    # ================================================================
    print("\n" + "=" * 80)
    print("(a) Individual LLM-CFD runs (unified scoring: top3_mean + p99_clip)")
    print("=" * 80)
    
    per_run = []
    per_run_auprc = []
    per_run_f1 = []
    best_auprc = -1.0
    best_run_scores = None
    best_run_idx = -1
    
    for i, run in enumerate(all_runs):
        validated = validate_cfds(df_clean, [c.copy() for c in run['cfds']])
        metrics = evaluate_discovery(validated, ADULT_GROUND_TRUTH)
        
        # UNIFIED: top3_mean + p99_clip (was score_mean)
        V = compute_violation_matrix(df_anom, validated, df_clean=df_clean)
        V_r, group_cfds = reduce_redundancy(V, validated)
        scores = apply_p99_clip(score_topk_mean(V_r, group_cfds, k=3))
        auprc = average_precision_score(labels, scores)
        
        per_run.append({
            'n_cfd': len(run['cfds']),
            'n_validated': len(validated),
            'n_groups': len(group_cfds),
            'f1': float(metrics['f1']),
            'auprc': float(auprc),
        })
        per_run_auprc.append(auprc)
        per_run_f1.append(metrics['f1'])
        
        if auprc > best_auprc:
            best_auprc = auprc
            best_run_scores = scores.copy()
            best_run_idx = i
        
        print(f"  Run {i+1:2d}: n_cfd={len(run['cfds']):2d} n_val={len(validated):2d} "
              f"n_grp={len(group_cfds):2d} F1={metrics['f1']:.3f} AUPRC={auprc:.4f}")
    
    f1_median = float(np.median(per_run_f1))
    f1_iqr = float(np.percentile(per_run_f1, 75) - np.percentile(per_run_f1, 25))
    auprc_median = float(np.median(per_run_auprc))
    auprc_iqr = float(np.percentile(per_run_auprc, 75) - np.percentile(per_run_auprc, 25))
    auprc_best = float(max(per_run_auprc))
    
    print(f"\n  F1:    median={f1_median:.3f} IQR={f1_iqr:.3f}")
    print(f"  AUPRC: median={auprc_median:.4f} IQR={auprc_iqr:.4f} best={auprc_best:.4f} (Run {best_run_idx+1})")
    
    # ================================================================
    # (b) HSCV fixed config (cat>=50%, top3_mean)
    # ================================================================
    print("\n" + "=" * 80)
    print("(b) HSCV fixed config (cat>=50%, top3_mean)")
    print("=" * 80)
    
    voted_cfds = hybrid_vote(all_runs, categorical_threshold=0.5)
    validated_hscv = validate_cfds(df_clean, [c.copy() for c in voted_cfds])
    metrics_hscv = evaluate_discovery(validated_hscv, ADULT_GROUND_TRUTH)
    
    V_hscv = compute_violation_matrix(df_anom, validated_hscv, df_clean=df_clean)
    V_r_hscv, group_cfds_hscv = reduce_redundancy(V_hscv, validated_hscv)
    hscv_scores = apply_p99_clip(score_topk_mean(V_r_hscv, group_cfds_hscv, k=3))
    hscv_auprc = average_precision_score(labels, hscv_scores)
    
    print(f"  Voted: {len(voted_cfds)}, Validated: {len(validated_hscv)}, Groups: {len(group_cfds_hscv)}")
    print(f"  F1={metrics_hscv['f1']:.3f} AUPRC={hscv_auprc:.4f}")
    
    # ================================================================
    # (c) Fusion: HSCV + COPOD
    # ================================================================
    print("\n" + "=" * 80)
    print("(c) HSCV + COPOD fusion")
    print("=" * 80)
    
    norm_hscv = normalize_scores(hscv_scores)
    norm_copod = normalize_scores(copod_scores)
    fusion_hscv_copod = (norm_hscv + norm_copod) / 2.0
    auprc_fusion = average_precision_score(labels, fusion_hscv_copod)
    print(f"  HSCV AUPRC: {hscv_auprc:.4f}, COPOD AUPRC: {auprc_copod:.4f}, Fusion AUPRC: {auprc_fusion:.4f}")
    
    # ================================================================
    # (d) CFDMiner-BL 4×4 sweep
    # ================================================================
    print("\n" + "=" * 80)
    print("(d) CFDMiner-BL 4×4 hyperparameter sweep")
    print("=" * 80)
    
    sweep_results, best_cfdminer = cfdminer_sweep_adult(df_clean)
    print(f"\n  Best CFDMiner-BL: sup={best_cfdminer['min_support']:.3f} conf={best_cfdminer['min_confidence']:.2f} "
          f"n={best_cfdminer['n_candidates']} F1={best_cfdminer['f1']:.3f}")
    
    # ================================================================
    # (e) PR curve data for figure generation
    # ================================================================
    print("\n" + "=" * 80)
    print("(e) Computing PR curve data")
    print("=" * 80)
    
    prec_hscv, rec_hscv, _ = precision_recall_curve(labels, hscv_scores)
    prec_eif, rec_eif, _ = precision_recall_curve(labels, normalize_scores(eif_scores))
    prec_copod, rec_copod, _ = precision_recall_curve(labels, normalize_scores(copod_scores))
    prec_fusion, rec_fusion, _ = precision_recall_curve(labels, fusion_hscv_copod)
    
    pr_data = {
        'hscv': {'precision': prec_hscv.tolist(), 'recall': rec_hscv.tolist(), 'auprc': float(hscv_auprc)},
        'eif': {'precision': prec_eif.tolist(), 'recall': rec_eif.tolist(), 'auprc': float(auprc_eif)},
        'copod': {'precision': prec_copod.tolist(), 'recall': rec_copod.tolist(), 'auprc': float(auprc_copod)},
        'fusion': {'precision': prec_fusion.tolist(), 'recall': rec_fusion.tolist(), 'auprc': float(auprc_fusion)},
        'anomaly_rate': float(labels.mean()),
    }
    
    # ================================================================
    # SAVE RESULTS
    # ================================================================
    output = {
        "description": "Adult unified experiment: same pipeline as Telco (top3_mean + p99_clip)",
        "dataset": "UCI Adult (10,000 rows sampled from 32,561, seed=42, 14 attributes)",
        "dataset_source": "Kohavi, R. (1996). Census Income [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5GP7S",
        "dataset_license": "CC BY 4.0",
        "scoring": "gradient + redundancy reduction + top3_mean + p99_clip",
        "n_gt_rules": len(ADULT_GROUND_TRUTH),
        "n_runs": n_runs,
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
            "n_groups": len(group_cfds_hscv),
            "f1": float(metrics_hscv['f1']),
            "auprc": float(hscv_auprc),
        },
        "baselines": {
            "eif_auprc": float(auprc_eif),
            "copod_auprc": float(auprc_copod),
        },
        "fusion": {
            "hscv_copod_auprc": float(auprc_fusion),
        },
        "cfdminer_bl": {
            "best_config": best_cfdminer,
            "all_configs": sweep_results,
            "n_gt_rules": len(ADULT_GROUND_TRUTH),
        },
        "pr_curve_data": pr_data,
    }
    
    output_file = os.path.join(RESULTS_PATH, "adult_unified_results.json")
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {output_file}")
    
    # ================================================================
    # SUMMARY
    # ================================================================
    print("\n" + "=" * 80)
    print("SUMMARY: Adult Unified Experiment")
    print("=" * 80)
    print(f"  Individual:  F1 median={f1_median:.3f}, AUPRC median={auprc_median:.4f}, best={auprc_best:.4f}")
    print(f"  HSCV:        F1={metrics_hscv['f1']:.3f}, AUPRC={hscv_auprc:.4f}")
    print(f"  EIF:         AUPRC={auprc_eif:.4f}")
    print(f"  COPOD:       AUPRC={auprc_copod:.4f}")
    print(f"  Fusion:      AUPRC={auprc_fusion:.4f}")
    print(f"  CFDMiner-BL: F1={best_cfdminer['f1']:.3f} (sup={best_cfdminer['min_support']:.3f}, conf={best_cfdminer['min_confidence']:.2f})")
    print(f"  GT rules: {len(ADULT_GROUND_TRUTH)}")
    print(f"  Improvement: HSCV vs median = +{(hscv_auprc - auprc_median)/max(auprc_median, 1e-8)*100:.1f}%")

if __name__ == "__main__":
    main()
