#!/usr/bin/env python3
"""Diagnostic: Compare unified v2 vs cross_llm pipeline on same cot_fewshot data."""
import json, os, sys, warnings
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, average_precision_score
warnings.filterwarnings('ignore')

sys.path.insert(0, "/sessions/69e790d3616697199023cd0a/workspace/mqw7dr6nxdtvfujn/experiments/src")

# Import both pipelines
from self_consistency_voting_v2 import (
    validate_cfds, compute_violation_matrix, reduce_redundancy,
    score_topk_mean, apply_p99_clip, inject_anomalies, load_data,
    evaluate_discovery, GROUND_TRUTH_CFDS, cfd_key, TYPE_MAP
)
from experiment import build_llm_prompt, parse_cfd_response
from cross_llm_experiment import (
    validate_cfds as validate_cross, compute_violation_matrix as compute_v_cross,
    reduce_redundancy as reduce_cross, score_topk_mean as score_cross,
    apply_p99_clip as clip_cross, inject_anomalies as inject_cross
)

CACHE_DIR = "/sessions/69e790d3616697199023cd0a/workspace/mqw7dr6nxdtvfujn/experiments/results/supplementary/cache_cot"
DATA_PATH = "/sessions/69e790d3616697199023cd0a/workspace/mqw7dr6nxdtvfujn/experiments/data/telco_churn.csv"

def main():
    # Load data
    df = pd.read_csv(DATA_PATH)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
    df = df.drop(columns=['customerID'])

    # Load first cot_fewshot cache
    cache_files = sorted([f for f in os.listdir(CACHE_DIR) if f.endswith('.json')])
    with open(os.path.join(CACHE_DIR, cache_files[0])) as f:
        response_text = json.load(f)

    # Parse CFDs
    valid_columns = set(df.columns)
    cfds = parse_cfd_response(response_text, valid_columns)
    print(f"Parsed CFDs: {len(cfds)}")
    print(f"CFD types: {[c.get('type') for c in cfds]}")

    # Inject anomalies (same seed)
    df_anom, labels = inject_anomalies(df, seed=42)
    df_clean = df.copy()
    print(f"Anomalies: {int(labels.sum())} / {len(labels)} = {labels.mean():.3f}")
    print(f"Anomaly types: dep_destroy={int(0.10*len(df))}, range={int(0.05*len(df))}, logic={int(0.03*len(df))}")

    # ===========================
    # PIPELINE 1: Unified v2
    # ===========================
    print("\n" + "="*60)
    print("PIPELINE 1: Unified v2")
    print("="*60)

    validated_v2 = validate_cfds(df_anom.copy(), cfds.copy())
    print(f"Validated CFDs (v2): {len(validated_v2)}")
    for c in validated_v2:
        print(f"  {c.get('type'):12s} dep={c.get('dependent_attribute'):20s} cond={c.get('condition_attributes')}")

    V_v2 = compute_violation_matrix(df_anom.copy(), validated_v2, df_clean=df_clean)
    print(f"\nV_v2 shape: {V_v2.shape}")
    print(f"V_v2 col means: {V_v2.mean(axis=0)}")
    print(f"V_v2 col max:   {V_v2.max(axis=0)}")
    print(f"V_v2 anomaly scores (mean): {V_v2[labels==1].mean(axis=0)}")
    print(f"V_v2 normal scores (mean):  {V_v2[labels==0].mean(axis=0)}")

    V_r_v2, gc_v2 = reduce_redundancy(V_v2, validated_v2)
    print(f"\nV_r_v2 shape: {V_r_v2.shape} (groups={len(gc_v2)})")

    raw_scores_v2 = score_topk_mean(V_r_v2, gc_v2, k=3)
    scores_v2 = apply_p99_clip(raw_scores_v2)
    auprc_v2 = average_precision_score(labels, scores_v2)
    print(f"\nAUPRC (v2): {auprc_v2:.4f}")
    print(f"Score stats: min={scores_v2.min():.4f} max={scores_v2.max():.4f} mean={scores_v2.mean():.4f}")
    print(f"Anomaly scores (mean): {scores_v2[labels==1].mean():.4f}")
    print(f"Normal scores (mean):  {scores_v2[labels==0].mean():.4f}")

    # ===========================
    # PIPELINE 2: Cross-LLM
    # ===========================
    print("\n" + "="*60)
    print("PIPELINE 2: Cross-LLM")
    print("="*60)

    validated_cross = validate_cross(df_anom.copy(), cfds.copy())
    print(f"Validated CFDs (cross): {len(validated_cross)}")
    for c in validated_cross:
        print(f"  {c.get('type'):12s} dep={c.get('dependent_attribute'):20s} cond={c.get('condition_attributes')}")

    V_cross = compute_v_cross(df_anom.copy(), validated_cross, df_clean=df_clean)
    print(f"\nV_cross shape: {V_cross.shape}")
    print(f"V_cross col means: {V_cross.mean(axis=0)}")
    print(f"V_cross col max:   {V_cross.max(axis=0)}")
    print(f"V_cross anomaly scores (mean): {V_cross[labels==1].mean(axis=0)}")
    print(f"V_cross normal scores (mean):  {V_cross[labels==0].mean(axis=0)}")

    V_r_cross, gc_cross = reduce_cross(V_cross, validated_cross)
    print(f"\nV_r_cross shape: {V_r_cross.shape} (groups={len(gc_cross)})")

    raw_scores_cross = score_cross(V_r_cross, gc_cross, k=3)
    scores_cross = clip_cross(raw_scores_cross)
    auprc_cross = average_precision_score(labels, scores_cross)
    print(f"\nAUPRC (cross): {auprc_cross:.4f}")
    print(f"Score stats: min={scores_cross.min():.4f} max={scores_cross.max():.4f} mean={scores_cross.mean():.4f}")
    print(f"Anomaly scores (mean): {scores_cross[labels==1].mean():.4f}")
    print(f"Normal scores (mean):  {scores_cross[labels==0].mean():.4f}")

    # ===========================
    # COMPARISON
    # ===========================
    print("\n" + "="*60)
    print("COMPARISON")
    print("="*60)
    print(f"Validated CFDs: v2={len(validated_v2)} vs cross={len(validated_cross)}")
    print(f"V matrix shape: v2={V_v2.shape} vs cross={V_cross.shape}")
    print(f"Groups: v2={len(gc_v2)} vs cross={len(gc_cross)}")
    print(f"AUPRC:  v2={auprc_v2:.4f} vs cross={auprc_cross:.4f}")

    # Check if validated CFDs are the same
    v2_keys = set(cfd_key(c) for c in validated_v2)
    cross_keys = set(cfd_key(c) for c in validated_cross)
    print(f"\nValidated CFD keys match: {v2_keys == cross_keys}")
    if v2_keys != cross_keys:
        print(f"  Only in v2: {v2_keys - cross_keys}")
        print(f"  Only in cross: {cross_keys - v2_keys}")

    # Compare V matrices column by column (if same shape)
    if V_v2.shape == V_cross.shape:
        for j in range(V_v2.shape[1]):
            diff = np.abs(V_v2[:, j] - V_cross[:, j])
            if diff.max() > 0.01:
                cfd = validated_v2[j] if j < len(validated_v2) else {}
                print(f"  Col {j} ({cfd.get('type','?')} {cfd.get('dependent_attribute','?')}): max_diff={diff.max():.4f}")

if __name__ == "__main__":
    main()
