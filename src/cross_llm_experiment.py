#!/usr/bin/env python3
"""
P2-1: Cross-LLM Model Comparison
=================================
Runs GLM-4 on the same Telco Churn dataset with the same prompt as DeepSeek,
then compares CFD discovery quality and anomaly detection performance.

Demonstrates that LLM-CFD is model-agnostic, not dependent on a specific LLM.
"""
import json, os, re, hashlib, time, warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score
from scipy import stats as sp_stats
from openai import OpenAI
warnings.filterwarnings('ignore')

# ============================================================
# CONFIG
# ============================================================
DATA_PATH = "/sessions/69e790d3616697199023cd0a/workspace/mqw7dr6nxdtvfujn/experiments/data/telco_churn.csv"
CACHE_DIR = "/sessions/69e790d3616697199023cd0a/workspace/mqw7dr6nxdtvfujn/experiments/results/cross_llm/cache"
RESULTS_PATH = "/sessions/69e790d3616697199023cd0a/workspace/mqw7dr6nxdtvfujn/experiments/results/cross_llm"
FIGURES_PATH = "/sessions/69e790d3616697199023cd0a/workspace/mqw7dr6nxdtvfujn/experiments/results/figures"
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(RESULTS_PATH, exist_ok=True)
os.makedirs(FIGURES_PATH, exist_ok=True)

# GLM API (Zhipu AI)
GLM_API_KEY = os.environ.get("GLM_API_KEY", "")
if not GLM_API_KEY:
    GLM_API_KEY = input("Please enter your GLM API key: ").strip()
    if not GLM_API_KEY:
        raise ValueError("GLM_API_KEY not set. Export it as environment variable or input when prompted.")
GLM_BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"
GLM_MODEL = "glm-4.7-flash"

# DeepSeek API (for comparison, already cached)
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
if not DEEPSEEK_API_KEY:
    DEEPSEEK_API_KEY = input("Please enter your DeepSeek API key: ").strip()
    if not DEEPSEEK_API_KEY:
        raise ValueError("DEEPSEEK_API_KEY not set. Export it as environment variable or input when prompted.")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"

# Experiment params (same as main experiment)
SUPPORT_THRESHOLD = 0.01
CONFIDENCE_THRESHOLDS = {"high": 0.90, "medium": 0.85, "low": 0.75}
RANDOM_SEED = 42
N_RUNS = 5  # 5 runs per LLM (enough for comparison, cost-efficient)

TYPE_MAP = {'fd': 'categorical', 'enum': 'categorical', 'logic': 'categorical',
            'range': 'numerical', 'consistency': 'structural'}
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

# ============================================================
# DATA LOADING (same as main experiment)
# ============================================================
def load_data():
    df = pd.read_csv(DATA_PATH)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
    df = df.drop(columns=['customerID'])
    return df

def inject_anomalies(df, seed=42):
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
# PROMPT (reuse from experiment.py)
# ============================================================
# Import the prompt builder from experiment.py
import sys
sys.path.insert(0, '/sessions/69e790d3616697199023cd0a/workspace/mqw7dr6nxdtvfujn/experiments/src')
from experiment import build_llm_prompt, parse_cfd_response

def parse_cfds(response_text, valid_columns):
    """Parse LLM response into CFD list, with truncation recovery."""
    # First try the standard parser from experiment.py
    cfds = parse_cfd_response(response_text, valid_columns)
    if cfds:
        return cfds

    # Fallback: handle truncated JSON by extracting complete top-level objects
    json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
    else:
        json_match = re.search(r'\[\s*\{.*', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
        else:
            return []

    # Track brace depth to find complete top-level objects (handles nested braces)
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

# ============================================================
# LLM API CALLS
# ============================================================
def call_glm(prompt, temperature=0.0):
    """Call GLM-4.7-Flash API with larger max_tokens to avoid truncation."""
    client = OpenAI(api_key=GLM_API_KEY, base_url=GLM_BASE_URL)
    response = client.chat.completions.create(
        model=GLM_MODEL,
        messages=[
            {"role": "system", "content": "你是一位数据质量专家。请严格按照要求输出JSON格式的结果，不要输出任何其他内容。"},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=8192,
    )
    return response.choices[0].message.content

def call_deepseek(prompt, temperature=0.0):
    """Call DeepSeek API."""
    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
    response = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[
            {"role": "system", "content": "你是一位数据质量专家。请严格按照要求输出JSON格式的结果，不要输出任何其他内容。"},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=4096,
    )
    return response.choices[0].message.content

def cache_filename(prompt, model_name, run_idx):
    h = hashlib.md5(f"{model_name}_{run_idx}_{prompt[:100]}".encode()).hexdigest()
    return f"{model_name}_{h}.json"

# ============================================================
# STATISTICAL VALIDATION (same as HSCV v2)
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
# ANOMALY SCORING (with gradient + redundancy from P0)
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
            # In-range distributional anomalies are still in the allowed set,
            # but they are rare -> low P -> high violation score
            cond_probs = _compute_conditional_prob(df_clean, dep, cond_attrs, cond_vals)
            for idx in sub_idx:
                actual = str(df.loc[idx, dep])
                p_actual = cond_probs.get(actual, 0.0)
                V[idx, j] = 1.0 - p_actual
        elif cfd_type == 'consistency':
            if dep == 'TotalCharges' and all(c in df.columns for c in ['tenure', 'MonthlyCharges']):
                expected_vals = df.loc[sub_idx, 'tenure'] * df.loc[sub_idx, 'MonthlyCharges']
                actual_vals = pd.to_numeric(df.loc[sub_idx, dep], errors='coerce')
                ratio = abs(actual_vals - expected_vals) / (expected_vals.abs() + 1e-8)
                V[sub_idx, j] = ratio.clip(0, 1).values
        elif cfd_type == 'logic':
            expr = ep.get('expression', '')
            if 'Churn' in df.columns:
                if 'tenure>24' in expr:
                    clean_mask = df_clean['tenure'] > 24
                    if clean_mask.sum() > 0:
                        p_churn_yes = (df_clean[clean_mask]['Churn'] == 'Yes').mean()
                    else:
                        p_churn_yes = 0.5
                    for idx in sub_idx:
                        if df.loc[idx, 'tenure'] > 24 and df.loc[idx, 'Churn'] == 'Yes':
                            V[idx, j] = 1.0 - p_churn_yes
                elif 'tenure<6' in expr:
                    clean_mask = df_clean['tenure'] < 6
                    if clean_mask.sum() > 0:
                        p_churn_no = (df_clean[clean_mask]['Churn'] == 'No').mean()
                    else:
                        p_churn_no = 0.5
                    for idx in sub_idx:
                        if df.loc[idx, 'tenure'] < 6 and df.loc[idx, 'Churn'] == 'No':
                            V[idx, j] = 1.0 - p_churn_no
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
        V_reduced[:, i] = V[:, info['cols']].max(axis=1)
        best = max(info['cfds'], key=lambda c: c.get('confidence', 0.5))
        best['n_merged_rules'] = len(info['cols'])
        group_cfds.append(best)
    return V_reduced, group_cfds

def score_topk_mean(V, cfds, k=3):
    if V.shape[1] == 0: return np.zeros(V.shape[0])
    k = min(k, V.shape[1])
    if k == 0: return np.zeros(V.shape[0])
    return np.sort(V, axis=1)[:, -k:].mean(axis=1)

def apply_p99_clip(scores):
    p99 = np.percentile(scores, 99)
    if p99 > 0:
        scores = np.clip(scores / p99, 0, 1)
    return scores

def evaluate_discovery(discovered, ground_truth):
    gt_keys = set(cfd_key(c) for c in ground_truth)
    disc_keys = set(cfd_key(c) for c in discovered)
    tp = len(gt_keys & disc_keys)
    fp = len(disc_keys - gt_keys)
    fn = len(gt_keys - disc_keys)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return {'precision': precision, 'recall': recall, 'f1': f1, 'tp': tp, 'fp': fp, 'fn': fn}

def compute_eif_scores(df_anom):
    df_enc = df_anom.copy()
    for col in df_enc.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df_enc[col] = le.fit_transform(df_enc[col].astype(str))
    iforest = IsolationForest(n_estimators=200, contamination=0.18, random_state=RANDOM_SEED)
    iforest.fit(df_enc)
    return -iforest.score_samples(df_enc)

# ============================================================
# MAIN
# ============================================================
def run_llm_experiment(model_name, call_fn, prompt, df, df_anom, df_clean, labels, valid_columns, n_runs):
    """Run n_runs of CFD discovery + anomaly detection for a given LLM."""
    eif_scores = compute_eif_scores(df_anom)
    auprc_eif = average_precision_score(labels, eif_scores)
    
    all_runs = []
    per_run = []
    
    for run_idx in range(n_runs):
        cache_file = os.path.join(CACHE_DIR, cache_filename(prompt, model_name, run_idx))
        
        if os.path.exists(cache_file):
            print(f"  {model_name} Run {run_idx+1}: Loading from cache")
            with open(cache_file) as f:
                response = f.read()
        else:
            print(f"  {model_name} Run {run_idx+1}: Calling API...")
            try:
                response = call_fn(prompt, temperature=0.0)
                with open(cache_file, 'w') as f:
                    f.write(response)
                print(f"    Cached to {os.path.basename(cache_file)}")
            except Exception as e:
                print(f"    API ERROR: {str(e)[:150]}")
                continue
            time.sleep(1)
        
        cfds = parse_cfds(response, valid_columns)
        validated = validate_cfds(df_clean, cfds)
        metrics = evaluate_discovery(validated, GROUND_TRUTH_CFDS)
        
        V = compute_violation_matrix(df_anom, validated, df_clean=df_clean)
        V_r, group_cfds = reduce_redundancy(V, validated)
        scores = apply_p99_clip(score_topk_mean(V_r, group_cfds))
        auprc = average_precision_score(labels, scores)
        
        has_consistency = any(c.get('type') == 'consistency' for c in validated)
        has_range_tenure = any(c.get('type') == 'range' and c.get('dependent_attribute') == 'tenure' for c in validated)
        
        all_runs.append({'cfds': cfds, 'validated': validated})
        per_run.append({
            'n_candidates': len(cfds),
            'n_validated': len(validated),
            'n_groups': len(group_cfds),
            'f1': metrics['f1'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'auprc': auprc,
            'has_consistency': has_consistency,
            'has_range_tenure': has_range_tenure,
        })
        
        print(f"    n_cfd={len(cfds):2d} n_val={len(validated):2d} "
              f"F1={metrics['f1']:.3f} AUPRC={auprc:.3f} "
              f"{'[cons]' if has_consistency else ''}{'[rng-t]' if has_range_tenure else ''}")
    
    f1s = [r['f1'] for r in per_run]
    auprcs = [r['auprc'] for r in per_run]
    
    return {
        'model': model_name,
        'n_runs': len(per_run),
        'per_run': per_run,
        'f1_median': float(np.median(f1s)),
        'f1_iqr': float(np.percentile(f1s, 75) - np.percentile(f1s, 25)),
        'auprc_median': float(np.median(auprcs)),
        'auprc_iqr': float(np.percentile(auprcs, 75) - np.percentile(auprcs, 25)),
        'auprc_best': float(max(auprcs)) if auprcs else 0,
        'n_runs_with_consistency': sum(1 for r in per_run if r['has_consistency']),
        'auprc_eif': float(auprc_eif),
        'all_runs': all_runs,
    }

def main():
    print("=" * 70)
    print("P2-1: Cross-LLM Model Comparison")
    print("  DeepSeek-V4-Flash (deepseek-chat) vs GLM-4.7-Flash")
    print("=" * 70)
    
    # Load data
    df = load_data()
    df_clean = df.copy()
    df_anom, labels = inject_anomalies(df, seed=RANDOM_SEED)
    print(f"Data: {df.shape}, Anomalies: {int(labels.sum())}")
    
    # Build prompt (same cot_fewshot strategy as main experiment)
    df_sample = df.sample(min(100, len(df)), random_state=RANDOM_SEED)
    prompt = build_llm_prompt(
        # Build schema metadata (reuse from experiment.py)
        [{'name': col, 'dtype': str(df[col].dtype),
          'type': 'numerical' if df[col].dtype in ['int64', 'float64'] else 'categorical',
          'min': float(df[col].min()) if df[col].dtype in ['int64', 'float64'] else None,
          'max': float(df[col].max()) if df[col].dtype in ['int64', 'float64'] else None,
          'mean': float(df[col].mean()) if df[col].dtype in ['int64', 'float64'] else None,
          'std': float(df[col].std()) if df[col].dtype in ['int64', 'float64'] else None,
          'nunique': int(df[col].nunique()) if df[col].dtype == 'object' else None,
          'values': df[col].value_counts().head(10).to_dict() if df[col].dtype == 'object' else None,
          'top_value': df[col].value_counts().index[0] if df[col].dtype == 'object' else None,
          'top_freq': float(df[col].value_counts().iloc[0] / len(df)) if df[col].dtype == 'object' else None,
          'samples': [str(s) for s in df[col].dropna().sample(min(8, len(df)), random_state=RANDOM_SEED).tolist()],
         } for col in df.columns],
        df_sample,
        strategy='cot_fewshot'
    )
    valid_columns = set(df.columns)
    
    # Run DeepSeek (5 runs)
    print(f"\n{'='*70}")
    print(f"Running DeepSeek ({N_RUNS} runs)")
    print(f"{'='*70}")
    deepseek_results = run_llm_experiment(
        'deepseek-chat', call_deepseek, prompt, df, df_anom, df_clean, labels, valid_columns, N_RUNS
    )
    
    # Run GLM (5 runs)
    print(f"\n{'='*70}")
    print(f"Running GLM-4.7-Flash ({N_RUNS} runs)")
    print(f"{'='*70}")
    glm_results = run_llm_experiment(
        'glm-4.7-flash', call_glm, prompt, df, df_anom, df_clean, labels, valid_columns, N_RUNS
    )
    
    # ================================================================
    # HSCV for each LLM (using fixed config: cat≥50%, top3_mean)
    # ================================================================
    print(f"\n{'='*70}")
    print("HSCV (Fixed Config: cat≥50%, top3_mean)")
    print(f"{'='*70}")
    
    for name, results in [('DeepSeek', deepseek_results), ('GLM', glm_results)]:
        all_runs = results['all_runs']
        n_runs = len(all_runs)
        
        # Hybrid vote
        cat_min_votes = int(np.ceil(n_runs * 0.5))
        key_to_info = {}
        for run_idx, run in enumerate(all_runs):
            seen = set()
            for cfd in run['cfds']:
                key = cfd_key_with_type(cfd)
                if key in seen: continue
                seen.add(key)
                if key not in key_to_info:
                    key_to_info[key] = {'cfds': [], 'count': 0, 'orig_type': cfd.get('type', 'fd')}
                key_to_info[key]['cfds'].append(cfd)
                key_to_info[key]['count'] += 1
        
        voted_cfds = []
        for key, info in key_to_info.items():
            is_cat = info['orig_type'] in CATEGORICAL_TYPES
            if (is_cat and info['count'] >= cat_min_votes) or (not is_cat and info['count'] >= 1):
                best = max(info['cfds'], key=lambda c: {'high':3,'medium':2,'low':1}.get(c.get('confidence_estimate','low'),0))
                voted_cfds.append(best)
        
        validated = validate_cfds(df_clean, voted_cfds)
        metrics = evaluate_discovery(validated, GROUND_TRUTH_CFDS)
        V = compute_violation_matrix(df_anom, validated, df_clean=df_clean)
        V_r, group_cfds = reduce_redundancy(V, validated)
        scores = apply_p99_clip(score_topk_mean(V_r, group_cfds))
        auprc = average_precision_score(labels, scores)
        
        has_cons = any(c.get('type') == 'consistency' for c in validated)
        print(f"  {name:12s}: n_voted={len(voted_cfds):2d} n_val={len(validated):2d} "
              f"F1={metrics['f1']:.3f} AUPRC={auprc:.3f} {'[cons]' if has_cons else ''}")
        
        results['hscv'] = {
            'n_voted': len(voted_cfds),
            'n_validated': len(validated),
            'f1': metrics['f1'],
            'auprc': float(auprc),
            'has_consistency': has_cons,
        }
    
    # ================================================================
    # Summary
    # ================================================================
    print(f"\n{'='*70}")
    print("SUMMARY: Cross-LLM Comparison")
    print(f"{'='*70}")
    print(f"\n{'Metric':<25} {'DeepSeek':>12} {'GLM-4.7-Flash':>15} {'EIF':>10}")
    print("-" * 62)
    print(f"{'F1 median':<25} {deepseek_results['f1_median']:>12.3f} {glm_results['f1_median']:>12.3f} {'—':>10}")
    print(f"{'F1 IQR':<25} {deepseek_results['f1_iqr']:>12.3f} {glm_results['f1_iqr']:>12.3f} {'—':>10}")
    print(f"{'AUPRC median':<25} {deepseek_results['auprc_median']:>12.3f} {glm_results['auprc_median']:>12.3f} {deepseek_results['auprc_eif']:>10.3f}")
    print(f"{'AUPRC IQR':<25} {deepseek_results['auprc_iqr']:>12.3f} {glm_results['auprc_iqr']:>12.3f} {'—':>10}")
    print(f"{'AUPRC best':<25} {deepseek_results['auprc_best']:>12.3f} {glm_results['auprc_best']:>12.3f} {'—':>10}")
    print(f"{'Runs w/ consistency':<25} {deepseek_results['n_runs_with_consistency']:>12} {glm_results['n_runs_with_consistency']:>12} {'—':>10}")
    print(f"{'HSCV AUPRC':<25} {deepseek_results['hscv']['auprc']:>12.3f} {glm_results['hscv']['auprc']:>12.3f} {'—':>10}")
    print(f"{'HSCV F1':<25} {deepseek_results['hscv']['f1']:>12.3f} {glm_results['hscv']['f1']:>12.3f} {'—':>10}")
    
    # Agreement analysis
    print(f"\n{'='*70}")
    print("CFD Discovery Agreement Analysis")
    print(f"{'='*70}")
    
    # Compare CFD sets
    ds_cfds = set()
    for run in deepseek_results['all_runs']:
        for c in run['validated']:
            ds_cfds.add(cfd_key(c))
    
    glm_cfds = set()
    for run in glm_results['all_runs']:
        for c in run['validated']:
            glm_cfds.add(cfd_key(c))
    
    gt_keys = set(cfd_key(c) for c in GROUND_TRUTH_CFDS)
    
    overlap = ds_cfds & glm_cfds
    ds_only = ds_cfds - glm_cfds
    glm_only = glm_cfds - ds_cfds
    
    print(f"  DeepSeek unique CFDs: {len(ds_cfds)}")
    print(f"  GLM unique CFDs:      {len(glm_cfds)}")
    print(f"  Overlap (both found): {len(overlap)}")
    print(f"  DeepSeek only:        {len(ds_only)}")
    print(f"  GLM only:             {len(glm_only)}")
    print(f"  Jaccard similarity:   {len(overlap) / len(ds_cfds | glm_cfds):.3f}")
    print(f"\n  Ground truth coverage:")
    print(f"    DeepSeek: {len(ds_cfds & gt_keys)}/{len(gt_keys)} = {len(ds_cfds & gt_keys)/len(gt_keys)*100:.0f}%")
    print(f"    GLM:      {len(glm_cfds & gt_keys)}/{len(gt_keys)} = {len(glm_cfds & gt_keys)/len(gt_keys)*100:.0f}%")
    print(f"    Both:     {len(overlap & gt_keys)}/{len(gt_keys)} = {len(overlap & gt_keys)/len(gt_keys)*100:.0f}%")
    
    # Save results
    # Remove all_runs (not JSON-serializable with CFD objects)
    ds_save = {k: v for k, v in deepseek_results.items() if k != 'all_runs'}
    glm_save = {k: v for k, v in glm_results.items() if k != 'all_runs'}
    
    output = {
        'description': 'Cross-LLM comparison: DeepSeek vs GLM-4.7-Flash on Telco Churn',
        'n_runs_per_llm': N_RUNS,
        'deepseek': ds_save,
        'glm': glm_save,
        'agreement': {
            'deepseek_unique': len(ds_cfds),
            'glm_unique': len(glm_cfds),
            'overlap': len(overlap),
            'deepseek_only': len(ds_only),
            'glm_only': len(glm_only),
            'jaccard': len(overlap) / len(ds_cfds | glm_cfds) if (ds_cfds | glm_cfds) else 0,
            'gt_coverage_deepseek': len(ds_cfds & gt_keys) / len(gt_keys),
            'gt_coverage_glm': len(glm_cfds & gt_keys) / len(gt_keys),
        },
        'eif_auprc': deepseek_results['auprc_eif'],
    }
    
    output_file = os.path.join(RESULTS_PATH, "cross_llm_results.json")
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    main()
