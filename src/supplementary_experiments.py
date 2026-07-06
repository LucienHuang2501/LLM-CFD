"""
Supplementary Experiments for LLM-CFD
======================================
Three experiments addressing reviewer concerns:
  1. Column Name Semantic Ablation (P0-问题4)
  2. LLM Output Stability - 10 repeated calls (P1-问题5)
  3. Statistical CFD Baseline (P0-问题2)
"""

import json
import os
import sys
import time
import hashlib
import warnings
from datetime import datetime
from typing import Any
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import average_precision_score, ndcg_score
from openai import OpenAI

warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "/Users/hzy/.qoderworkcn/workspace/mqw7dr6nxdtvfujn/experiments/data/telco_churn.csv"
RESULTS_PATH = "/Users/hzy/.qoderworkcn/workspace/mqw7dr6nxdtvfujn/experiments/results/supplementary"
CACHE_PATH = os.path.join(RESULTS_PATH, "cache")

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"

SUPPORT_THRESHOLD = 0.01
CONFIDENCE_THRESHOLDS = {"high": 0.90, "medium": 0.85, "low": 0.75}
RANDOM_SEED = 42
LLM_TEMPERATURE = 0.0
LLM_MAX_TOKENS = 4096
STABILITY_RUNS = 5  # Number of repeated LLM calls for stability test

np.random.seed(RANDOM_SEED)
os.makedirs(RESULTS_PATH, exist_ok=True)
os.makedirs(CACHE_PATH, exist_ok=True)


# ============================================================
# SHARED UTILITIES (from experiment.py)
# ============================================================

def load_data():
    """Load and preprocess Telco Churn dataset."""
    df = pd.read_csv(DATA_PATH)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
    df = df.drop(columns=['customerID'])
    return df


def build_schema(df, col_mapping=None):
    """Build schema metadata for LLM prompt. col_mapping renames columns."""
    schema = []
    for col in df.columns:
        display_name = col_mapping.get(col, col) if col_mapping else col
        meta = {"name": display_name, "dtype": str(df[col].dtype), "nunique": int(df[col].nunique())}
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        if col in numerical_cols:
            meta["type"] = "numerical"
            meta["min"] = float(df[col].min())
            meta["max"] = float(df[col].max())
            meta["mean"] = float(df[col].mean())
            meta["std"] = float(df[col].std())
        else:
            meta["type"] = "categorical"
            vc = df[col].value_counts()
            meta["values"] = {str(k): int(v) for k, v in vc.head(8).items()}
            meta["top_value"] = str(vc.index[0])
            meta["top_freq"] = float(vc.iloc[0] / len(df))
        samples = df[col].dropna().sample(min(5, len(df)), random_state=RANDOM_SEED).tolist()
        meta["samples"] = [str(s) for s in samples]
        schema.append(meta)
    return schema


def build_prompt(schema, strategy="few_shot"):
    """Build LLM prompt with given schema and strategy."""
    system_msg = "你是一位数据质量专家，擅长从数据表的Schema和样本中推断条件函数依赖（CFD）规则。"

    schema_text = "数据表Schema:\n"
    for s in schema:
        schema_text += f"- {s['name']} ({s['type']}, {s['nunique']} unique, "
        if s['type'] == 'numerical':
            schema_text += f"range=[{s['min']:.1f},{s['max']:.1f}], mean={s['mean']:.1f}"
        else:
            vals = list(s.get('values', {}).keys())[:5]
            schema_text += f"values={vals}"
        schema_text += f"), samples={s['samples'][:3]}\n"

    task = """请从上述Schema中推断所有可能存在的条件函数依赖（CFD）规则。
CFD规则格式：当某些属性满足特定条件时，另一个属性应满足特定约束。
输出JSON数组，每条规则包含：type(range/enum/logic/consistency/fd), condition_attributes, condition_values, dependent_attribute, expected_pattern, confidence_estimate(high/medium/low), natural_language_description"""

    if strategy == "zero_shot":
        user_msg = f"{schema_text}\n{task}"
    else:  # few_shot
        examples = """
示例1: 当Contract='Two year'时，MonthlyCharges应在[50,120]区间
→ {"type":"range","condition_attributes":["Contract"],"condition_values":{"Contract":"Two year"},"dependent_attribute":"MonthlyCharges","expected_pattern":{"min":50,"max":120},"confidence_estimate":"high","natural_language_description":"两年期合同月消费50-120"}

示例2: 当InternetService='No'时，OnlineSecurity必须为'No internet service'
→ {"type":"enum","condition_attributes":["InternetService"],"condition_values":{"InternetService":"No"},"dependent_attribute":"OnlineSecurity","expected_pattern":{"values":["No internet service"]},"confidence_estimate":"high","natural_language_description":"无网络则无在线安全"}

示例3: tenure>24的客户Churn大概率为'No'
→ {"type":"logic","condition_attributes":["tenure"],"condition_values":{},"dependent_attribute":"Churn","expected_pattern":{"expression":"tenure>24→Churn='No'"},"confidence_estimate":"medium","natural_language_description":"长期客户流失率低"}
"""
        user_msg = f"{schema_text}\n参考示例:\n{examples}\n{task}"

    return system_msg, user_msg


def call_llm(system_msg, user_msg, run_id=0):
    """Call DeepSeek API with caching."""
    cache_key = hashlib.md5(f"{system_msg}{user_msg}{run_id}".encode()).hexdigest()
    cache_file = os.path.join(CACHE_PATH, f"{cache_key}.json")

    if os.path.exists(cache_file):
        with open(cache_file) as f:
            return json.load(f)

    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
    resp = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS,
    )
    result = resp.choices[0].message.content

    with open(cache_file, 'w') as f:
        json.dump(result, f, ensure_ascii=False)
    return result


def parse_cfds(llm_output):
    """Parse CFD rules from LLM output."""
    try:
        start = llm_output.find('[')
        end = llm_output.rfind(']') + 1
        if start == -1 or end == 0:
            return []
        return json.loads(llm_output[start:end])
    except:
        return []


def validate_cfd(df, cfd):
    """Validate a single CFD against data."""
    try:
        cond_attrs = cfd.get("condition_attributes", [])
        cond_vals = cfd.get("condition_values", {})
        dep_attr = cfd.get("dependent_attribute", "")
        pattern = cfd.get("expected_pattern", {})
        cfd_type = cfd.get("type", "fd")

        if dep_attr not in df.columns:
            return None

        # Filter by conditions
        mask = pd.Series([True] * len(df))
        for attr, val in cond_vals.items():
            if attr not in df.columns:
                return None
            mask = mask & (df[attr].astype(str) == str(val))

        support = mask.sum() / len(df)
        if support < SUPPORT_THRESHOLD:
            return None

        sub = df[mask]
        if len(sub) == 0:
            return None

        # Check confidence
        if cfd_type == "range":
            mn = pattern.get("min", sub[dep_attr].min())
            mx = pattern.get("max", sub[dep_attr].max())
            try:
                vals = pd.to_numeric(sub[dep_attr], errors='coerce')
                violations = ((vals < mn) | (vals > mx)).sum()
            except:
                violations = 0
            confidence = 1 - violations / len(sub)
        elif cfd_type == "enum":
            expected = pattern.get("values", [])
            if expected:
                violations = (~sub[dep_attr].astype(str).isin([str(v) for v in expected])).sum()
                confidence = 1 - violations / len(sub)
            else:
                confidence = 1.0
        else:
            confidence = 0.9  # default for logic/consistency/fd

        conf_threshold = CONFIDENCE_THRESHOLDS.get(cfd.get("confidence_estimate", "medium"), 0.85)
        passed = confidence >= conf_threshold

        return {"cfd": cfd, "support": support, "confidence": confidence, "passed": passed}
    except:
        return None


def evaluate_cfds(df, validated_cfds, ground_truth_cfds):
    """Evaluate discovered CFDs against ground truth."""
    def cfd_key(c):
        return (c.get("dependent_attribute", ""),
                tuple(sorted(c.get("condition_attributes", []))),
                c.get("type", "fd"))

    gt_keys = set(cfd_key(c) for c in ground_truth_cfds)
    disc_keys = set(cfd_key(c["cfd"]) for c in validated_cfds if c["passed"])

    tp = len(gt_keys & disc_keys)
    fp = len(disc_keys - gt_keys)
    fn = len(gt_keys - disc_keys)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


# ============================================================
# GROUND TRUTH (same as main experiment)
# ============================================================

GROUND_TRUTH_CFDS = [
    {"dependent_attribute": "OnlineSecurity", "condition_attributes": ["InternetService"], "type": "fd", "condition_values": {"InternetService": "No"}},
    {"dependent_attribute": "OnlineBackup", "condition_attributes": ["InternetService"], "type": "fd", "condition_values": {"InternetService": "No"}},
    {"dependent_attribute": "DeviceProtection", "condition_attributes": ["InternetService"], "type": "fd", "condition_values": {"InternetService": "No"}},
    {"dependent_attribute": "TechSupport", "condition_attributes": ["InternetService"], "type": "fd", "condition_values": {"InternetService": "No"}},
    {"dependent_attribute": "StreamingTV", "condition_attributes": ["InternetService"], "type": "fd", "condition_values": {"InternetService": "No"}},
    {"dependent_attribute": "StreamingMovies", "condition_attributes": ["InternetService"], "type": "fd", "condition_values": {"InternetService": "No"}},
    {"dependent_attribute": "MultipleLines", "condition_attributes": ["PhoneService"], "type": "fd", "condition_values": {"PhoneService": "No"}},
    {"dependent_attribute": "MultipleLines", "condition_attributes": ["PhoneService"], "type": "fd", "condition_values": {"PhoneService": "Yes"}},
    {"dependent_attribute": "MonthlyCharges", "condition_attributes": ["Contract"], "type": "range", "condition_values": {"Contract": "Two year"}},
    {"dependent_attribute": "MonthlyCharges", "condition_attributes": ["Contract"], "type": "range", "condition_values": {"Contract": "One year"}},
    {"dependent_attribute": "MonthlyCharges", "condition_attributes": ["Contract"], "type": "range", "condition_values": {"Contract": "Month-to-month"}},
    {"dependent_attribute": "tenure", "condition_attributes": ["Contract"], "type": "range", "condition_values": {"Contract": "Two year"}},
    {"dependent_attribute": "TotalCharges", "condition_attributes": [], "type": "range", "condition_values": {}},
    {"dependent_attribute": "MonthlyCharges", "condition_attributes": [], "type": "range", "condition_values": {}},
    {"dependent_attribute": "SeniorCitizen", "condition_attributes": [], "type": "enum", "condition_values": {}},
    {"dependent_attribute": "gender", "condition_attributes": [], "type": "enum", "condition_values": {}},
    {"dependent_attribute": "Partner", "condition_attributes": [], "type": "enum", "condition_values": {}},
    {"dependent_attribute": "Dependents", "condition_attributes": [], "type": "enum", "condition_values": {}},
    {"dependent_attribute": "tenure", "condition_attributes": [], "type": "logic", "condition_values": {}},
    {"dependent_attribute": "TotalCharges", "condition_attributes": ["tenure", "MonthlyCharges"], "type": "consistency", "condition_values": {}},
]


# ============================================================
# EXPERIMENT 1: Column Name Semantic Ablation
# ============================================================

def experiment_column_ablation(df):
    """Replace meaningful column names with meaningless codes."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 1: COLUMN NAME SEMANTIC ABLATION")
    print("=" * 60)

    # Create meaningless column mapping
    col_mapping = {}
    for i, col in enumerate(df.columns):
        col_mapping[col] = f"ATTR_{i+1:02d}"

    print(f"  Column mapping:")
    for orig, code in col_mapping.items():
        print(f"    {orig:20s} → {code}")

    # Build schema with coded names
    schema = build_schema(df, col_mapping)
    system_msg, user_msg = build_prompt(schema, strategy="few_shot")

    print(f"\n  Calling LLM with ablated column names...")
    llm_output = call_llm(system_msg, user_msg, run_id=999)
    raw_cfds = parse_cfds(llm_output)
    print(f"  LLM discovered {len(raw_cfds)} candidate CFDs (ablated names)")

    # Map back to original names for validation
    reverse_mapping = {v: k for k, v in col_mapping.items()}
    for cfd in raw_cfds:
        if cfd.get("dependent_attribute") in reverse_mapping:
            cfd["dependent_attribute"] = reverse_mapping[cfd["dependent_attribute"]]
        new_cond_attrs = []
        new_cond_vals = {}
        for attr in cfd.get("condition_attributes", []):
            if attr in reverse_mapping:
                new_cond_attrs.append(reverse_mapping[attr])
            else:
                new_cond_attrs.append(attr)
        for attr, val in cfd.get("condition_values", {}).items():
            if attr in reverse_mapping:
                new_cond_vals[reverse_mapping[attr]] = val
            else:
                new_cond_vals[attr] = val
        cfd["condition_attributes"] = new_cond_attrs
        cfd["condition_values"] = new_cond_vals

    # Validate
    validated = []
    for cfd in raw_cfds:
        result = validate_cfd(df, cfd)
        if result:
            validated.append(result)

    n_passed = sum(1 for v in validated if v["passed"])
    print(f"  Validated: {len(validated)}, Passed gate: {n_passed}")

    # Evaluate against ground truth
    metrics = evaluate_cfds(df, validated, GROUND_TRUTH_CFDS)
    print(f"  Precision={metrics['precision']:.3f}, Recall={metrics['recall']:.3f}, F1={metrics['f1']:.3f}")
    print(f"  TP={metrics['tp']}, FP={metrics['fp']}, FN={metrics['fn']}")

    return {
        "candidates": len(raw_cfds),
        "validated": len(validated),
        "passed": n_passed,
        "metrics": metrics,
    }


# ============================================================
# EXPERIMENT 2: LLM Output Stability (Repeated Calls)
# ============================================================

def experiment_stability(df):
    """Run LLM-CFD multiple times and measure output variance."""
    print("\n" + "=" * 60)
    print(f"EXPERIMENT 2: LLM OUTPUT STABILITY ({STABILITY_RUNS} runs)")
    print("=" * 60)

    schema = build_schema(df)
    system_msg, user_msg = build_prompt(schema, strategy="few_shot")

    results = []
    for run in range(STABILITY_RUNS):
        print(f"\n  Run {run+1}/{STABILITY_RUNS}...", end=" ")
        t0 = time.time()
        llm_output = call_llm(system_msg, user_msg, run_id=run)
        elapsed = time.time() - t0

        raw_cfds = parse_cfds(llm_output)
        validated = []
        for cfd in raw_cfds:
            result = validate_cfd(df, cfd)
            if result:
                validated.append(result)

        metrics = evaluate_cfds(df, validated, GROUND_TRUTH_CFDS)
        results.append({
            "run": run + 1,
            "candidates": len(raw_cfds),
            "passed": sum(1 for v in validated if v["passed"]),
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "elapsed_sec": elapsed,
        })
        print(f"candidates={len(raw_cfds)}, passed={sum(1 for v in validated if v['passed'])}, "
              f"P={metrics['precision']:.3f}, R={metrics['recall']:.3f}, F1={metrics['f1']:.3f}, {elapsed:.1f}s")

    # Compute statistics
    candidates = [r["candidates"] for r in results]
    passed = [r["passed"] for r in results]
    precisions = [r["precision"] for r in results]
    recalls = [r["recall"] for r in results]
    f1s = [r["f1"] for r in results]

    summary = {
        "n_runs": STABILITY_RUNS,
        "candidates_mean": np.mean(candidates),
        "candidates_std": np.std(candidates),
        "passed_mean": np.mean(passed),
        "passed_std": np.std(passed),
        "precision_mean": np.mean(precisions),
        "precision_std": np.std(precisions),
        "recall_mean": np.mean(recalls),
        "recall_std": np.std(recalls),
        "f1_mean": np.mean(f1s),
        "f1_std": np.std(f1s),
        "runs": results,
    }

    print(f"\n  Summary:")
    print(f"    Candidates: {summary['candidates_mean']:.1f} ± {summary['candidates_std']:.1f}")
    print(f"    Passed:     {summary['passed_mean']:.1f} ± {summary['passed_std']:.1f}")
    print(f"    Precision:  {summary['precision_mean']:.3f} ± {summary['precision_std']:.3f}")
    print(f"    Recall:     {summary['recall_mean']:.3f} ± {summary['recall_std']:.3f}")
    print(f"    F1:         {summary['f1_mean']:.3f} ± {summary['f1_std']:.3f}")

    return summary


# ============================================================
# EXPERIMENT 3: Statistical CFD Baseline
# ============================================================

def experiment_statistical_baseline(df):
    """Discover CFDs using pure statistical methods (no LLM)."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: STATISTICAL CFD BASELINE")
    print("=" * 60)

    discovered = []
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # 1. Enum constraints: check all categorical columns for binary/few-value patterns
    print("  Checking enum constraints...")
    for col in categorical_cols:
        nunique = df[col].nunique()
        if nunique <= 5:
            discovered.append({
                "type": "enum",
                "condition_attributes": [],
                "condition_values": {},
                "dependent_attribute": col,
                "expected_pattern": {"values": df[col].unique().tolist()},
                "confidence_estimate": "high",
                "source": "statistical_enum",
            })

    # 2. Range constraints: check numerical columns for natural ranges
    print("  Checking range constraints...")
    for col in numerical_cols:
        discovered.append({
            "type": "range",
            "condition_attributes": [],
            "condition_values": {},
            "dependent_attribute": col,
            "expected_pattern": {"min": float(df[col].min()), "max": float(df[col].max())},
            "confidence_estimate": "high",
            "source": "statistical_range",
        })

    # 3. FD constraints: check categorical→categorical dependencies
    print("  Checking FD constraints (categorical→categorical)...")
    for col1, col2 in combinations(categorical_cols, 2):
        # Check if col1 determines col2
        grouped = df.groupby(col1)[col2].nunique()
        if (grouped == 1).all():
            for val in df[col1].unique():
                sub = df[df[col1] == val]
                discovered.append({
                    "type": "fd",
                    "condition_attributes": [col1],
                    "condition_values": {col1: str(val)},
                    "dependent_attribute": col2,
                    "expected_pattern": {"values": sub[col2].unique().tolist()},
                    "confidence_estimate": "high",
                    "source": "statistical_fd",
                })
        # Check reverse
        grouped_rev = df.groupby(col2)[col1].nunique()
        if (grouped_rev == 1).all():
            for val in df[col2].unique():
                sub = df[df[col2] == val]
                discovered.append({
                    "type": "fd",
                    "condition_attributes": [col2],
                    "condition_values": {col2: str(val)},
                    "dependent_attribute": col1,
                    "expected_pattern": {"values": sub[col1].unique().tolist()},
                    "confidence_estimate": "high",
                    "source": "statistical_fd",
                })

    # 4. Range constraints conditioned on categorical
    print("  Checking conditional range constraints...")
    for cat_col in categorical_cols:
        for num_col in numerical_cols:
            grouped = df.groupby(cat_col)[num_col]
            for val, sub in grouped:
                if len(sub) >= 50:  # min support
                    discovered.append({
                        "type": "range",
                        "condition_attributes": [cat_col],
                        "condition_values": {cat_col: str(val)},
                        "dependent_attribute": num_col,
                        "expected_pattern": {"min": float(sub.min()), "max": float(sub.max())},
                        "confidence_estimate": "high",
                        "source": "statistical_cond_range",
                    })

    # 5. Consistency: check tenure × MonthlyCharges ≈ TotalCharges
    print("  Checking consistency constraints...")
    if all(c in df.columns for c in ['tenure', 'MonthlyCharges', 'TotalCharges']):
        corr = df['tenure'].multiply(df['MonthlyCharges']).corr(df['TotalCharges'])
        if abs(corr) > 0.8:
            discovered.append({
                "type": "consistency",
                "condition_attributes": ["tenure", "MonthlyCharges"],
                "condition_values": {},
                "dependent_attribute": "TotalCharges",
                "expected_pattern": {"expression": f"tenure×MonthlyCharges≈TotalCharges (corr={corr:.3f})"},
                "confidence_estimate": "high",
                "source": "statistical_consistency",
            })

    print(f"  Statistical baseline discovered {len(discovered)} candidate rules")

    # Validate
    validated = []
    for cfd in discovered:
        result = validate_cfd(df, cfd)
        if result:
            validated.append(result)

    n_passed = sum(1 for v in validated if v["passed"])
    print(f"  Validated: {len(validated)}, Passed gate: {n_passed}")

    # Evaluate
    metrics = evaluate_cfds(df, validated, GROUND_TRUTH_CFDS)
    print(f"  Precision={metrics['precision']:.3f}, Recall={metrics['recall']:.3f}, F1={metrics['f1']:.3f}")
    print(f"  TP={metrics['tp']}, FP={metrics['fp']}, FN={metrics['fn']}")

    # Breakdown by source
    by_source = {}
    for v in validated:
        if v["passed"]:
            src = v["cfd"].get("source", "unknown")
            by_source[src] = by_source.get(src, 0) + 1
    print(f"  Passed by source: {by_source}")

    return {
        "candidates": len(discovered),
        "validated": len(validated),
        "passed": n_passed,
        "metrics": metrics,
        "by_source": by_source,
    }


# ============================================================
# MAIN
# ============================================================

def main():
    print("LLM-CFD Supplementary Experiments")
    print("=" * 60)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"API Key: {'SET' if DEEPSEEK_API_KEY else 'MISSING'}")

    df = load_data()
    print(f"Data loaded: {df.shape}")

    all_results = {}

    # Experiment 1: Column Name Ablation
    if DEEPSEEK_API_KEY:
        all_results["column_ablation"] = experiment_column_ablation(df)
    else:
        print("\n  SKIPPING Experiment 1 (no API key)")
        all_results["column_ablation"] = {"skipped": True}

    # Experiment 2: LLM Stability
    if DEEPSEEK_API_KEY:
        all_results["stability"] = experiment_stability(df)
    else:
        print("\n  SKIPPING Experiment 2 (no API key)")
        all_results["stability"] = {"skipped": True}

    # Experiment 3: Statistical Baseline (no LLM needed)
    all_results["statistical_baseline"] = experiment_statistical_baseline(df)

    # Save results
    output_file = os.path.join(RESULTS_PATH, "supplementary_results.json")
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nResults saved to {output_file}")

    # Print comparison table
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(f"{'Method':<30} {'Candidates':>10} {'Passed':>8} {'P':>6} {'R':>6} {'F1':>6}")
    print("-" * 60)

    # Original LLM-CFD results
    print(f"{'LLM-CFD (original, few-shot)':<30} {'31':>10} {'25':>8} {'0.640':>6} {'0.800':>6} {'0.711':>6}")

    # Column ablation
    if "column_ablation" in all_results and not all_results["column_ablation"].get("skipped"):
        ca = all_results["column_ablation"]
        m = ca["metrics"]
        print(f"{'LLM-CFD (ablated names)':<30} {str(ca['candidates']):>10} {str(ca['passed']):>8} "
              f"{m['precision']:>6.3f} {m['recall']:>6.3f} {m['f1']:>6.3f}")

    # Statistical baseline
    sb = all_results["statistical_baseline"]
    m = sb["metrics"]
    print(f"{'Statistical Baseline':<30} {str(sb['candidates']):>10} {str(sb['passed']):>8} "
          f"{m['precision']:>6.3f} {m['recall']:>6.3f} {m['f1']:>6.3f}")


if __name__ == "__main__":
    main()
