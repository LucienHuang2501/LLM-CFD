#!/usr/bin/env python3
"""
E2: CFDMiner-BL Hyperparameter Sweep
Sweeps min_support and min_confidence to find the best configuration for CFDMiner-BL.
Addresses Nature reviewer R1's concern about baseline fairness (confidence=0.90 may be too strict).
"""
import json, os, sys, warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from itertools import combinations
warnings.filterwarnings('ignore')

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = "/sessions/69e790d3616697199023cd0a/workspace/mqw7dr6nxdtvfujn/experiments/data/telco_churn.csv"
RESULTS_PATH = "/sessions/69e790d3616697199023cd0a/workspace/mqw7dr6nxdtvfujn/experiments/results/ctane"
os.makedirs(RESULTS_PATH, exist_ok=True)

# Ground truth (same as p0_revision_experiments.py)
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

TYPE_MAP = {'fd': 'categorical', 'enum': 'categorical', 'logic': 'categorical',
            'range': 'numerical', 'consistency': 'structural'}

def cfd_key(c):
    raw_type = c.get("type", "fd")
    normalized_type = TYPE_MAP.get(raw_type, raw_type)
    return (c.get("dependent_attribute", ""),
            tuple(sorted(c.get("condition_attributes", []))),
            normalized_type)

def evaluate(discovered, ground_truth):
    gt_keys = set(cfd_key(c) for c in ground_truth)
    disc_keys = set(cfd_key(c) for c in discovered)
    tp = len(gt_keys & disc_keys)
    fp = len(disc_keys - gt_keys)
    fn = len(gt_keys - disc_keys)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return {'precision': precision, 'recall': recall, 'f1': f1, 'tp': tp, 'fp': fp, 'fn': fn}

def cfdminer_baseline(df, min_support=0.01, min_confidence=0.90):
    candidates = []
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    df_enc = df.copy()
    for col in categorical_cols:
        le = LabelEncoder()
        df_enc[col] = le.fit_transform(df[col].astype(str))
    
    # Single attribute conditions only (for performance - pair-wise is too slow for sweep)
    condition_patterns = []
    for col in categorical_cols:
        for val in df[col].unique():
            mask = df[col] == val
            support = mask.sum() / len(df)
            if support >= min_support:
                condition_patterns.append({'attrs': [col], 'mask': mask, 'support': support})
    
    # Skip pair-wise conditions in sweep for performance
    # (pair-wise patterns are the same as in the full CFDMiner-BL, just too slow for 16 configs)
    
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
    
    # Range constraints (unconditional)
    for col in numerical_cols:
        vals = df[col].dropna()
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
    
    # Consistency
    if all(c in df.columns for c in ['tenure', 'MonthlyCharges', 'TotalCharges']):
        corr = (df['tenure'] * df['MonthlyCharges']).corr(df['TotalCharges'])
        if abs(corr) > 0.8:
            candidates.append({
                'type': 'consistency', 'condition_attributes': ['tenure', 'MonthlyCharges'],
                'dependent_attribute': 'TotalCharges', 'confidence': abs(corr), 'support': 1.0,
            })
    
    # Deduplicate
    seen = {}
    for c in candidates:
        key = (c['dependent_attribute'], tuple(sorted(c['condition_attributes'])), c['type'])
        if key not in seen or c['confidence'] > seen[key]['confidence']:
            seen[key] = c
    
    return list(seen.values())

def main():
    print("=" * 60)
    print("E2: CFDMiner-BL Hyperparameter Sweep")
    print("=" * 60)
    
    df = pd.read_csv(DATA_PATH)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
    df = df.drop(columns=['customerID'])
    print(f"Data: {df.shape}")
    
    # Parameter grid
    support_list = [0.005, 0.01, 0.02, 0.05]
    confidence_list = [0.80, 0.85, 0.90, 0.95]
    
    results = []
    
    for min_sup in support_list:
        for min_conf in confidence_list:
            candidates = cfdminer_baseline(df, min_support=min_sup, min_confidence=min_conf)
            metrics = evaluate(candidates, GROUND_TRUTH_CFDS)
            
            result = {
                'min_support': min_sup,
                'min_confidence': min_conf,
                'n_candidates': len(candidates),
                **metrics
            }
            results.append(result)
            print(f"  sup={min_sup:.3f} conf={min_conf:.2f} → n={len(candidates):4d} P={metrics['precision']:.3f} R={metrics['recall']:.3f} F1={metrics['f1']:.3f}")
    
    # Find best
    best = max(results, key=lambda x: x['f1'])
    print(f"\nBest config: sup={best['min_support']:.3f} conf={best['min_confidence']:.2f} → F1={best['f1']:.3f}")
    print(f"  (Original paper used sup=0.01 conf=0.90 → F1={next(r['f1'] for r in results if r['min_support']==0.01 and r['min_confidence']==0.90):.3f})")
    
    # LLM-CFD reference
    print(f"  LLM-CFD F1=0.711")
    if best['f1'] > 0:
        print(f"  LLM-CFD improvement over best CFDMiner-BL: {0.711 / best['f1']:.1f}x")
    
    # Save
    output = {
        'description': 'CFDMiner-BL hyperparameter sweep: 4x4 grid of (min_support, min_confidence)',
        'llm_cfd_f1': 0.711,
        'best_config': best,
        'all_configs': results,
        'note': 'Type normalization applied (fd/enum/logic→categorical, range→numerical, consistency→structural)'
    }
    
    output_file = os.path.join(RESULTS_PATH, "cfdminer_sweep_results.json")
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    main()
