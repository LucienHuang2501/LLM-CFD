#!/usr/bin/env python3
"""
HSCV Ablation & Temperature Experiment
=======================================
Part 1: Voting mechanism ablation (majority-only vs union-only vs HSCV hybrid)
        using the 10+1 cached LLM responses at temp=0.0 (NO API calls).

Part 2: Temperature experiment (0.0, 0.3, 0.5, 0.7) with 5 DeepSeek runs each.
        For temp=0.0, reuses existing cache when available.

Usage:
    cd experiments
    python3 src/hscv_ablation_temperature.py
"""
import json
import os
import re
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from openai import OpenAI

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
    validate_cfds,
    cfd_key,
    cfd_key_with_type,
    GROUND_TRUTH_CFDS,
    TYPE_MAP,
    CATEGORICAL_TYPES,
    CACHE_DIR,
)

# Imports from experiment.py
from experiment import build_llm_prompt, parse_cfd_response

# Imports from unified_experiment.py
from unified_experiment import (
    load_data,
    run_copod,
    evaluate_discovery,
    load_cached_response,
    parse_cfds_robust,
)

# compute_eif_scores is defined in self_consistency_voting_v2, not in
# unified_experiment; fall back gracefully.
try:
    from unified_experiment import compute_eif_scores
except ImportError:
    from self_consistency_voting_v2 import compute_eif_scores


# ============================================================
# CONFIG
# ============================================================
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
if not DEEPSEEK_API_KEY:
    DEEPSEEK_API_KEY = input("Please enter your DeepSeek API key: ").strip()
    if not DEEPSEEK_API_KEY:
        raise ValueError("DEEPSEEK_API_KEY not set. Export it as environment variable or input when prompted.")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"

TEMP_CACHE_DIR = "/sessions/69e790d3616697199023cd0a/workspace/mqw7dr6nxdtvfujn/experiments/results/hscv_temperature/cache"
TEMP_RESULTS_DIR = "/sessions/69e790d3616697199023cd0a/workspace/mqw7dr6nxdtvfujn/experiments/results/hscv_temperature"
RESULTS_FILE = os.path.join(
    TEMP_RESULTS_DIR, "hscv_ablation_temperature_results.json"
)

RANDOM_SEED = 42
N_RUNS_TEMP = 5
TEMPS = [0.0, 0.3, 0.5, 0.7]
MAX_TOKENS = 4096

os.makedirs(TEMP_CACHE_DIR, exist_ok=True)
os.makedirs(TEMP_RESULTS_DIR, exist_ok=True)


# ============================================================
# VOTING FUNCTION (adapted from cross_llm_experiment.py)
# ============================================================
def hybrid_vote(all_runs_cfds, cat_threshold=0.5, mode='hscv'):
    """Vote across runs.

    Parameters
    ----------
    all_runs_cfds : list[list[dict]]
        CFD lists from each LLM run.
    cat_threshold : float
        Fraction of runs required for categorical rules (default 0.5).
    mode : str
        'hscv'           — categorical ≥ threshold, numerical/structural ≥ 1
        'majority_only'  — ALL rule types require ≥ threshold
        'union_only'     — ALL rule types require ≥ 1

    Returns
    -------
    list[dict]
        Voted CFDs (best representative from each surviving key).
    """
    n_runs = len(all_runs_cfds)
    cat_min_votes = int(np.ceil(n_runs * cat_threshold))

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

    voted = []
    for key, info in key_to_info.items():
        is_cat = info['orig_type'] in CATEGORICAL_TYPES
        if mode == 'hscv':
            keep = (is_cat and info['count'] >= cat_min_votes) or \
                   (not is_cat and info['count'] >= 1)
        elif mode == 'majority_only':
            keep = info['count'] >= cat_min_votes
        elif mode == 'union_only':
            keep = info['count'] >= 1
        else:
            keep = False

        if keep:
            best = max(
                info['cfds'],
                key=lambda c: {'high': 3, 'medium': 2, 'low': 1}.get(
                    c.get('confidence_estimate', 'low'), 0
                ),
            )
            voted.append(best)

    return voted


# ============================================================
# UNIFIED PIPELINE SCORE
# ============================================================
def unified_pipeline_score(df_anom, validated_cfds, df_clean):
    """Score via unified pipeline:

    compute_violation_matrix → reduce_redundancy → score_topk_mean → apply_p99_clip

    Returns
    -------
    scores : np.ndarray
    n_groups : int
    """
    V = compute_violation_matrix(df_anom, validated_cfds, df_clean=df_clean)
    V_r, group_cfds = reduce_redundancy(V, validated_cfds)
    scores = apply_p99_clip(score_topk_mean(V_r, group_cfds))
    return scores, len(group_cfds)


# ============================================================
# DEEPSEEK API CALL
# ============================================================
def call_deepseek(prompt, temperature=0.0):
    """Call DeepSeek API with specified temperature."""
    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
    response = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[
            {
                "role": "system",
                "content": "你是一位数据质量专家。请严格按照要求输出JSON格式的结果，不要输出任何其他内容。",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=MAX_TOKENS,
    )
    return response.choices[0].message.content


# ============================================================
# SCHEMA METADATA BUILDER
# ============================================================
def build_schema_metadata(df):
    """Build schema metadata list for build_llm_prompt.

    Mirrors the construction in experiment.py's load_and_preprocess
    and cross_llm_experiment.py's inline approach.
    """
    schema_metadata = []
    for col in df.columns:
        meta = {
            "name": col,
            "dtype": str(df[col].dtype),
            "nunique": int(df[col].nunique()),
        }
        if df[col].dtype in ['int64', 'float64']:
            meta["type"] = "numerical"
            meta["min"] = float(df[col].min())
            meta["max"] = float(df[col].max())
            meta["mean"] = float(df[col].mean())
            meta["std"] = float(df[col].std())
        else:
            meta["type"] = "categorical"
            vc = df[col].value_counts()
            meta["values"] = vc.head(10).to_dict()
            meta["top_value"] = vc.index[0]
            meta["top_freq"] = float(vc.iloc[0] / len(df))
        samples = df[col].dropna().sample(
            min(8, len(df)), random_state=RANDOM_SEED
        ).tolist()
        meta["samples"] = [str(s) for s in samples]
        schema_metadata.append(meta)
    return schema_metadata


# ============================================================
# PART 1: VOTING MECHANISM ABLATION (no API calls)
# ============================================================
def run_ablation(df_clean, df_anom, labels, valid_columns):
    """Part 1: Ablation of voting mechanisms using cached responses."""
    print("=" * 80)
    print("PART 1: HSCV Voting Mechanism Ablation (cached 10+1 runs)")
    print("=" * 80)

    # Load all cached responses from CACHE_DIR (supplementary/cache)
    cache_files = sorted([f for f in os.listdir(CACHE_DIR) if f.endswith('.json')])
    print(f"Found {len(cache_files)} cache files in CACHE_DIR")

    all_runs_cfds = []
    for cf in cache_files:
        response = load_cached_response(os.path.join(CACHE_DIR, cf))
        cfds = parse_cfds_robust(response, valid_columns)
        all_runs_cfds.append(cfds)
        print(f"  {cf}: {len(cfds)} CFDs parsed")

    n_runs = len(all_runs_cfds)
    print(f"Total runs loaded: {n_runs}")

    ablation_results = {
        "description": "Voting mechanism ablation using 10+1 cached runs at temp=0.0",
    }

    for mode in ['majority_only', 'union_only', 'hscv']:
        print(f"\n--- Voting mode: {mode} ---")

        # Vote across all runs
        voted = hybrid_vote(all_runs_cfds, cat_threshold=0.5, mode=mode)
        print(f"  Voted CFDs: {len(voted)}")

        # Validate voted rules on clean data (use copies to preserve originals)
        validated = validate_cfds(df_clean, [c.copy() for c in voted])
        print(f"  Validated: {len(validated)}")

        # F1 via evaluate_discovery
        metrics = evaluate_discovery(validated, GROUND_TRUTH_CFDS)

        # Unified pipeline score → AUPRC
        scores, n_groups = unified_pipeline_score(df_anom, validated, df_clean)
        scores = np.nan_to_num(scores, nan=0.0)
        auprc = average_precision_score(labels, scores)

        print(f"  n_voted={len(voted)}  n_validated={len(validated)}  "
              f"n_groups={n_groups}")
        print(f"  F1={metrics['f1']:.4f}  AUPRC={auprc:.4f}")

        ablation_results[mode] = {
            "n_voted": len(voted),
            "n_validated": len(validated),
            "n_groups": n_groups,
            "f1": float(metrics['f1']),
            "auprc": float(auprc),
        }

    return {"ablation": ablation_results}


# ============================================================
# PART 2: TEMPERATURE EXPERIMENT (needs API calls)
# ============================================================
def run_temperature_experiment(df_clean, df_anom, labels, valid_columns, prompt):
    """Part 2: Temperature experiment with DeepSeek API (5 runs per temp)."""
    print("\n" + "=" * 80)
    print("PART 2: Temperature Experiment (DeepSeek, 5 runs each)")
    print("=" * 80)

    temp_results = {
        "description": "HSCV at different temperatures (5 runs each)",
        "temps": TEMPS,
        "results": {},
    }

    for temp in TEMPS:
        print(f"\n--- Temperature: {temp} ---")
        temp_key = f"{temp}"

        all_runs_cfds = []
        per_run_auprc = []
        per_run_has_consistency = []

        for run_idx in range(N_RUNS_TEMP):
            cache_file = os.path.join(
                TEMP_CACHE_DIR,
                f"deepseek-chat_temp{temp}_run{run_idx}.json",
            )

            # ── Check temp cache first ──
            if os.path.exists(cache_file):
                print(f"  Run {run_idx + 1}: Loading from temp cache")
                response = load_cached_response(cache_file)
            else:
                # ── For temp=0.0, try to reuse supplementary cache ──
                if temp == 0.0:
                    supp_files = sorted(
                        [f for f in os.listdir(CACHE_DIR) if f.endswith('.json')]
                    )
                    if run_idx < len(supp_files):
                        supp_path = os.path.join(CACHE_DIR, supp_files[run_idx])
                        print(f"  Run {run_idx + 1}: Reusing supplementary cache "
                              f"({supp_files[run_idx]})")
                        response = load_cached_response(supp_path)
                        # Persist to temp cache for future runs
                        with open(cache_file, 'w') as f:
                            f.write(response)
                    else:
                        # Not enough supplementary files — call API
                        print(f"  Run {run_idx + 1}: Calling DeepSeek API "
                              f"(temp={temp})...")
                        try:
                            response = call_deepseek(prompt, temperature=temp)
                            with open(cache_file, 'w') as f:
                                f.write(response)
                        except Exception as e:
                            print(f"    API ERROR: {str(e)[:150]}")
                            continue
                        time.sleep(1)
                else:
                    # ── Non-zero temp: call API ──
                    print(f"  Run {run_idx + 1}: Calling DeepSeek API "
                          f"(temp={temp})...")
                    try:
                        response = call_deepseek(prompt, temperature=temp)
                        with open(cache_file, 'w') as f:
                            f.write(response)
                    except Exception as e:
                        print(f"    API ERROR: {str(e)[:150]}")
                        continue
                    time.sleep(1)

            # Parse CFDs (with truncation fallback)
            cfds = parse_cfds_robust(response, valid_columns)
            all_runs_cfds.append(cfds)

            # Validate on clean data
            validated = validate_cfds(df_clean, [c.copy() for c in cfds])

            # Unified pipeline score → AUPRC
            scores, n_groups = unified_pipeline_score(df_anom, validated, df_clean)
            scores = np.nan_to_num(scores, nan=0.0)
            auprc = average_precision_score(labels, scores)

            has_consistency = bool(
                any(c.get('type') == 'consistency' for c in validated)
            )

            per_run_auprc.append(auprc)
            per_run_has_consistency.append(has_consistency)

            print(f"    n_cfd={len(cfds):2d}  n_val={len(validated):2d}  "
                  f"n_grp={n_groups:2d}  AUPRC={auprc:.4f}  "
                  f"{'[cons]' if has_consistency else ''}")

        # ── Individual run statistics ──
        if per_run_auprc:
            individual_median = float(np.median(per_run_auprc))
            individual_iqr = float(
                np.percentile(per_run_auprc, 75) - np.percentile(per_run_auprc, 25)
            )
        else:
            individual_median = 0.0
            individual_iqr = 0.0

        # ── Rule diversity: unique rules across all runs ──
        unique_keys = set()
        for cfds in all_runs_cfds:
            for cfd in cfds:
                unique_keys.add(cfd_key_with_type(cfd))
        n_unique_rules = len(unique_keys)

        # ── HSCV across the 5 runs (cat≥50%, top3_mean) ──
        voted = hybrid_vote(all_runs_cfds, cat_threshold=0.5, mode='hscv')
        validated_hscv = validate_cfds(df_clean, [c.copy() for c in voted])
        hscv_scores, hscv_n_groups = unified_pipeline_score(
            df_anom, validated_hscv, df_clean
        )
        hscv_scores = np.nan_to_num(hscv_scores, nan=0.0)
        hscv_auprc = average_precision_score(labels, hscv_scores)

        n_runs_with_consistency = sum(per_run_has_consistency)

        print(f"\n  Summary for temp={temp}:")
        print(f"    Individual median AUPRC: {individual_median:.4f}  "
              f"(IQR: {individual_iqr:.4f})")
        print(f"    HSCV AUPRC:              {hscv_auprc:.4f}  "
              f"(n_voted={len(voted)}, n_val={len(validated_hscv)})")
        print(f"    Unique rules (diversity): {n_unique_rules}")
        print(f"    Runs with consistency:    "
              f"{n_runs_with_consistency}/{N_RUNS_TEMP}")

        temp_results["results"][temp_key] = {
            "individual_median_auprc": individual_median,
            "individual_iqr": individual_iqr,
            "hscv_auprc": float(hscv_auprc),
            "n_unique_rules": n_unique_rules,
            "n_runs_with_consistency": n_runs_with_consistency,
        }

    return {"temperature": temp_results}


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 80)
    print("HSCV Ablation & Temperature Experiment")
    print("  Part 1: Voting mechanism ablation (cached, no API)")
    print("  Part 2: Temperature sweep (DeepSeek API, 5 runs/temp)")
    print("=" * 80)

    # ─── Load data ───
    df = load_data()
    df_clean = df.copy()
    df_anom, labels = inject_anomalies(df, seed=RANDOM_SEED)
    valid_columns = set(df.columns)
    print(f"\nData: {df.shape}, Anomalies: {int(labels.sum())} "
          f"({labels.mean() * 100:.1f}%)")

    # ─── Part 1: Ablation ───
    ablation_output = run_ablation(df_clean, df_anom, labels, valid_columns)

    # ─── Build prompt for Part 2 ───
    schema_metadata = build_schema_metadata(df)
    df_sample = df.sample(min(100, len(df)), random_state=RANDOM_SEED)
    prompt = build_llm_prompt(schema_metadata, df_sample, strategy='cot_fewshot')

    # ─── Part 2: Temperature experiment ───
    temp_output = run_temperature_experiment(
        df_clean, df_anom, labels, valid_columns, prompt
    )

    # ─── Combine and save results ───
    output = {**ablation_output, **temp_output}

    with open(RESULTS_FILE, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {RESULTS_FILE}")

    # ================================================================
    # SUMMARY TABLE
    # ================================================================
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)

    # Part 1
    print("\n--- Part 1: Voting Mechanism Ablation ---")
    header = (f"{'Mode':<18} {'n_voted':>8} {'n_validated':>12} "
              f"{'n_groups':>10} {'F1':>8} {'AUPRC':>10}")
    print(header)
    print("-" * len(header))
    for mode in ['majority_only', 'union_only', 'hscv']:
        r = output['ablation'][mode]
        print(f"{mode:<18} {r['n_voted']:>8} {r['n_validated']:>12} "
              f"{r['n_groups']:>10} {r['f1']:>8.4f} {r['auprc']:>10.4f}")

    # Part 2
    print("\n--- Part 2: Temperature Experiment ---")
    header2 = (f"{'Temp':<8} {'Ind.Median':>11} {'Ind.IQR':>9} "
               f"{'HSCV AUPRC':>12} {'UniqueRules':>12} {'Cons.Runs':>10}")
    print(header2)
    print("-" * len(header2))
    for temp in TEMPS:
        r = output['temperature']['results'][f"{temp}"]
        print(f"{temp:<8} {r['individual_median_auprc']:>11.4f} "
              f"{r['individual_iqr']:>9.4f} {r['hscv_auprc']:>12.4f} "
              f"{r['n_unique_rules']:>12} "
              f"{r['n_runs_with_consistency']:>10}")

    print("\n" + "=" * 80)
    print("Done.")
    print("=" * 80)


if __name__ == "__main__":
    main()
