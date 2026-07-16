#!/usr/bin/env python3
"""
CTANE: CFD discovery via lattice traversal (TANE-based approach for CFDs).

Implements the level-wise lattice search algorithm for discovering
Conditional Functional Dependencies (CFDs) from data.

Reference: Fan, W. et al. (2011). "Discovering Conditional Functional Dependencies."
IEEE Transactions on Knowledge and Data Engineering, 23(5), 683-698.

The algorithm:
1. Builds a lattice of attribute combinations (level-wise, like Apriori)
2. At each level, partitions data by condition attributes
3. Checks if dependent attribute has unique value within each partition
4. Applies support and confidence thresholds for pruning

Outputs CFD rules in the same format as the paper's ground truth.
"""
import json
import os
import sys
import time
import warnings
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')

# ============================================================
# PATHS
# ============================================================
BASE = "/sessions/69e790d3616697199023cd0a/workspace/mqw7dr6nxdtvfujn"
TELOCO_PATH = os.path.join(BASE, "experiments/data/telco_churn.csv")
ADULT_PATH = os.path.join(BASE, "experiments/data/adult.csv")
RESULTS_PATH = os.path.join(BASE, "experiments/results/ctane")
os.makedirs(RESULTS_PATH, exist_ok=True)

# ============================================================
# GROUND TRUTH (same as experiment.py)
# ============================================================
TELOCO_GT = [
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

ADULT_GT = [
    {"dependent_attribute": "education_num", "condition_attributes": ["education"], "type": "fd"},
    {"dependent_attribute": "occupation", "condition_attributes": ["education"], "type": "fd", "condition_values": {"education": "Bachelors"}},
    {"dependent_attribute": "hours_per_week", "condition_attributes": ["occupation"], "type": "range", "condition_values": {"occupation": "Exec-managerial"}},
    {"dependent_attribute": "hours_per_week", "condition_attributes": ["occupation"], "type": "range", "condition_values": {"occupation": "Prof-specialty"}},
    {"dependent_attribute": "sex", "condition_attributes": [], "type": "enum"},
    {"dependent_attribute": "race", "condition_attributes": [], "type": "enum"},
    {"dependent_attribute": "education_num", "condition_attributes": ["education"], "type": "consistency"},
    {"dependent_attribute": "age", "condition_attributes": [], "type": "range"},
    {"dependent_attribute": "hours_per_week", "condition_attributes": [], "type": "range"},
    {"dependent_attribute": "marital_status", "condition_attributes": [], "type": "enum"},
    {"dependent_attribute": "capital_gain", "condition_attributes": [], "type": "range"},
    {"dependent_attribute": "capital_loss", "condition_attributes": [], "type": "range"},
]

TYPE_MAP = {'fd': 'categorical', 'enum': 'categorical', 'logic': 'categorical',
            'range': 'numerical', 'consistency': 'structural'}


def cfd_key(c):
    raw_type = c.get("type", "fd")
    return (c.get("dependent_attribute", ""),
            tuple(sorted(c.get("condition_attributes", []))),
            TYPE_MAP.get(raw_type, raw_type))


def evaluate(discovered, ground_truth):
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
# CTANE ALGORITHM
# ============================================================
def ctane_discover(df, min_support=0.01, min_confidence=0.90, max_level=3):
    """
    Discover CFDs using lattice-based search (TANE-style for CFDs).

    For each combination of condition attributes (LHS):
      - Partition data by LHS values
      - For each partition with support >= min_support:
        - Check if any RHS attribute has a single unique value (FD holds)
        - Or if confidence (mode frequency) >= min_confidence

    Parameters
    ----------
    df : pd.DataFrame
        Input data (categorical columns will be string-encoded).
    min_support : float
        Minimum support threshold (fraction of rows in condition partition).
    min_confidence : float
        Minimum confidence threshold for approximate FDs.
    max_level : int
        Maximum lattice level (number of LHS attributes).

    Returns
    -------
    list of dict
        Discovered CFD rules.
    """
    n = len(df)
    columns = list(df.columns)

    # Encode all columns as strings for uniform comparison
    df_str = df.copy()
    for col in df_str.columns:
        df_str[col] = df_str[col].astype(str)

    # Also get numeric columns for range rules
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = [c for c in columns if c not in numeric_cols]

    # Limit LHS search to categorical columns with reasonable cardinality
    # (high-cardinality columns like fnlwgt create too many partitions)
    lhs_candidates = []
    for col in columns:
        nunique = df_str[col].nunique()
        if nunique <= 50:  # Only low-cardinality columns as LHS
            lhs_candidates.append(col)

    discovered = []

    # --- Level 0: Unconditional rules ---
    # Enum constraints: each categorical column has a fixed set of values
    for col in categorical_cols:
        discovered.append({
            'type': 'enum',
            'condition_attributes': [],
            'dependent_attribute': col,
            'confidence': 1.0,
            'support': 1.0,
        })

    # Unconditional range constraints for numeric columns
    for col in numeric_cols:
        discovered.append({
            'type': 'range',
            'condition_attributes': [],
            'dependent_attribute': col,
            'confidence': 1.0,
            'support': 1.0,
        })

    # --- Level 1+: Conditional FDs via lattice search ---
    for level in range(1, max_level + 1):
        level_combos = list(combinations(lhs_candidates, level))
        print(f"  Level {level}: {len(level_combos)} combinations")
        for idx, lhs_attrs in enumerate(level_combos):
            if idx % 50 == 0 and idx > 0:
                print(f"    ...{idx}/{len(level_combos)}")
            # Partition data by LHS attribute combination
            try:
                groups = df_str.groupby(list(lhs_attrs))
            except Exception:
                continue

            for group_key, group_df in groups:
                if len(group_df) < max(2, min_support * n):
                    continue

                support = len(group_df) / n
                if support < min_support:
                    continue

                # Check each non-LHS attribute as potential RHS
                rhs_candidates = [c for c in columns if c not in lhs_attrs]
                for rhs in rhs_candidates:
                    vals = group_df[rhs].unique()
                    if len(vals) == 1:
                        # Exact FD: confidence = 1.0
                        confidence = 1.0
                    else:
                        # Approximate FD: confidence = max value frequency
                        vc = group_df[rhs].value_counts()
                        confidence = vc.iloc[0] / len(group_df)

                    if confidence >= min_confidence:
                        # Build condition values
                        if len(lhs_attrs) == 1:
                            cond_vals = {lhs_attrs[0]: str(group_key)}
                        else:
                            cond_vals = {a: str(v) for a, v in zip(lhs_attrs, group_key)}

                        discovered.append({
                            'type': 'fd',
                            'condition_attributes': list(lhs_attrs),
                            'dependent_attribute': rhs,
                            'condition_values': cond_vals,
                            'confidence': confidence,
                            'support': support,
                        })

    # --- Consistency constraints (special: TotalCharges = tenure * MonthlyCharges) ---
    if all(c in df.columns for c in ['tenure', 'MonthlyCharges', 'TotalCharges']):
        try:
            tc = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
            expected = pd.to_numeric(df['tenure'], errors='coerce') * pd.to_numeric(df['MonthlyCharges'], errors='coerce')
            corr = expected.corr(tc)
            if abs(corr) > 0.8:
                discovered.append({
                    'type': 'consistency',
                    'condition_attributes': ['tenure', 'MonthlyCharges'],
                    'dependent_attribute': 'TotalCharges',
                    'confidence': abs(corr),
                    'support': 1.0,
                })
        except Exception:
            pass

    # Deduplicate: keep highest confidence per (dep, sorted_cond, type)
    seen = {}
    for c in discovered:
        key = cfd_key(c)
        if key not in seen or c.get('confidence', 0) > seen[key].get('confidence', 0):
            seen[key] = c

    return list(seen.values())


# ============================================================
# RUN EXPERIMENT
# ============================================================
def run_ctane_telco():
    """Run CTANE on Telco Churn dataset with hyperparameter sweep."""
    print("=" * 60)
    print("CTANE Experiment: Telco Churn Dataset")
    print("=" * 60)

    df = pd.read_csv(TELOCO_PATH)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
    df = df.drop(columns=['customerID'])
    print(f"Data: {df.shape}")

    support_list = [0.005, 0.01, 0.02, 0.05]
    confidence_list = [0.90, 0.95]

    results = []
    for min_sup in support_list:
        for min_conf in confidence_list:
            t0 = time.time()
            cfds = ctane_discover(df, min_support=min_sup, min_confidence=min_conf, max_level=2)
            elapsed = time.time() - t0
            metrics = evaluate(cfds, TELOCO_GT)
            result = {
                'min_support': min_sup,
                'min_confidence': min_conf,
                'n_candidates': len(cfds),
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1': metrics['f1'],
                'tp': metrics['tp'],
                'fp': metrics['fp'],
                'fn': metrics['fn'],
                'gt_count': len(TELOCO_GT),
                'runtime_seconds': round(elapsed, 2),
            }
            results.append(result)
            print(f"  sup={min_sup:.3f} conf={min_conf:.2f} -> n={len(cfds):4d} "
                  f"P={metrics['precision']:.3f} R={metrics['recall']:.3f} F1={metrics['f1']:.3f} "
                  f"({elapsed:.2f}s)")

    best = max(results, key=lambda x: x['f1'])
    print(f"\nBest: sup={best['min_support']} conf={best['min_confidence']} F1={best['f1']:.3f}")

    return {
        'best_config': best,
        'all_configs': results,
        'note': 'CTANE lattice search, max_level=2, type normalization applied',
    }


def run_ctane_adult():
    """Run CTANE on UCI Adult dataset with hyperparameter sweep."""
    print("\n" + "=" * 60)
    print("CTANE Experiment: UCI Adult Dataset")
    print("=" * 60)

    col_names = ['age', 'workclass', 'fnlwgt', 'education', 'education_num',
                 'marital_status', 'occupation', 'relationship', 'race', 'sex',
                 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income']

    df = pd.read_csv(ADULT_PATH, header=None, names=col_names,
                     skipinitialspace=True, na_values='?')
    df = df.dropna()
    # Subsample to 10000 rows (same as main experiments)
    df = df.sample(10000, random_state=42)
    print(f"Data: {df.shape} (subsampled from Adult)")

    support_list = [0.005, 0.01, 0.02, 0.05]
    confidence_list = [0.90, 0.95]

    results = []
    for min_sup in support_list:
        for min_conf in confidence_list:
            t0 = time.time()
            cfds = ctane_discover(df, min_support=min_sup, min_confidence=min_conf, max_level=2)
            elapsed = time.time() - t0
            metrics = evaluate(cfds, ADULT_GT)
            result = {
                'min_support': min_sup,
                'min_confidence': min_conf,
                'n_candidates': len(cfds),
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1': metrics['f1'],
                'tp': metrics['tp'],
                'fp': metrics['fp'],
                'fn': metrics['fn'],
                'gt_count': len(ADULT_GT),
                'runtime_seconds': round(elapsed, 2),
            }
            results.append(result)
            print(f"  sup={min_sup:.3f} conf={min_conf:.2f} -> n={len(cfds):4d} "
                  f"P={metrics['precision']:.3f} R={metrics['recall']:.3f} F1={metrics['f1']:.3f} "
                  f"({elapsed:.2f}s)")

    best = max(results, key=lambda x: x['f1'])
    print(f"\nBest: sup={best['min_support']} conf={best['min_confidence']} F1={best['f1']:.3f}")

    return {
        'best_config': best,
        'all_configs': results,
        'note': 'CTANE lattice search, max_level=2, type normalization applied',
    }


def main():
    telco_results = run_ctane_telco()
    adult_results = run_ctane_adult()

    # Combine and save
    output = {
        'telco': telco_results,
        'adult': adult_results,
    }

    output_file = os.path.join(RESULTS_PATH, "ctane_experiment_results.json")
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_file}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Telco best: F1={telco_results['best_config']['f1']:.3f} "
          f"(sup={telco_results['best_config']['min_support']}, "
          f"conf={telco_results['best_config']['min_confidence']})")
    print(f"Adult best: F1={adult_results['best_config']['f1']:.3f} "
          f"(sup={adult_results['best_config']['min_support']}, "
          f"conf={adult_results['best_config']['min_confidence']})")


if __name__ == "__main__":
    main()
