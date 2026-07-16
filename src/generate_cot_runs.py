#!/usr/bin/env python3
"""
Generate 15 LLM runs using cot_fewshot strategy at temp=0.0.
Uses the same build_llm_prompt as cross_llm_experiment.py for consistency.

Saves to: results/supplementary/cache_cot/
"""
import json, os, sys, hashlib, time, warnings
import pandas as pd
from openai import OpenAI
warnings.filterwarnings('ignore')

# Import build_llm_prompt from experiment.py (same as cross_llm_experiment.py)
sys.path.insert(0, "/sessions/69e790d3616697199023cd0a/workspace/mqw7dr6nxdtvfujn/experiments/src")
from experiment import build_llm_prompt, parse_cfd_response

# ============================================================
# CONFIG
# ============================================================
DATA_PATH = "/sessions/69e790d3616697199023cd0a/workspace/mqw7dr6nxdtvfujn/experiments/data/telco_churn.csv"
CACHE_DIR = "/sessions/69e790d3616697199023cd0a/workspace/mqw7dr6nxdtvfujn/experiments/results/supplementary/cache_cot"
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"
LLM_TEMPERATURE = 0.0
LLM_MAX_TOKENS = 4096
RANDOM_SEED = 42
N_RUNS = 15

os.makedirs(CACHE_DIR, exist_ok=True)

# ============================================================
# DATA LOADING (same as cross_llm_experiment.py)
# ============================================================
def load_data():
    df = pd.read_csv(DATA_PATH)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
    df = df.drop(columns=['customerID'])
    return df

def build_schema_metadata(df):
    """Build schema metadata same as cross_llm_experiment.py."""
    schema = []
    for col in df.columns:
        is_num = df[col].dtype in ['int64', 'float64']
        meta = {
            'name': col,
            'dtype': str(df[col].dtype),
            'type': 'numerical' if is_num else 'categorical',
            'min': float(df[col].min()) if is_num else None,
            'max': float(df[col].max()) if is_num else None,
            'mean': float(df[col].mean()) if is_num else None,
            'std': float(df[col].std()) if is_num else None,
            'nunique': int(df[col].nunique()) if df[col].dtype == 'object' else None,
            'values': df[col].value_counts().head(10).to_dict() if df[col].dtype == 'object' else None,
            'top_value': df[col].value_counts().index[0] if df[col].dtype == 'object' else None,
            'top_freq': float(df[col].value_counts().iloc[0] / len(df)) if df[col].dtype == 'object' else None,
            'samples': [str(s) for s in df[col].dropna().sample(min(8, len(df)), random_state=RANDOM_SEED).tolist()],
        }
        schema.append(meta)
    return schema

def main():
    print("=" * 70)
    print(f"GENERATING {N_RUNS} COT_FEWSHOT RUNS (temp={LLM_TEMPERATURE})")
    print("=" * 70)

    if not DEEPSEEK_API_KEY:
        print("ERROR: DEEPSEEK_API_KEY not set!")
        sys.exit(1)

    df = load_data()
    schema_metadata = build_schema_metadata(df)
    df_sample = df.sample(min(100, len(df)), random_state=RANDOM_SEED)

    # Build prompt using cot_fewshot strategy (same as cross_llm_experiment.py)
    prompt = build_llm_prompt(schema_metadata, df_sample, strategy='cot_fewshot')
    valid_columns = set(df.columns)

    print(f"Prompt length: {len(prompt)} chars")
    print(f"Cache dir: {CACHE_DIR}")

    # Check existing files
    existing = [f for f in os.listdir(CACHE_DIR) if f.endswith('.json')]
    print(f"Existing cache files: {len(existing)}")

    # Generate 15 runs
    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

    run_results = []
    for run_id in range(N_RUNS):
        cache_file = os.path.join(CACHE_DIR, f"cot_run_{run_id:02d}.json")

        if os.path.exists(cache_file):
            print(f"  Run {run_id}: Already cached, loading...", end=" ", flush=True)
            with open(cache_file) as f:
                result = json.load(f)
        else:
            print(f"  Run {run_id}: Calling DeepSeek API...", end=" ", flush=True)
            t0 = time.time()
            resp = client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=[
                    {"role": "system", "content": "你是一位数据质量专家。请严格按照要求输出JSON格式的结果，不要输出任何其他内容。"},
                    {"role": "user", "content": prompt},
                ],
                temperature=LLM_TEMPERATURE,
                max_tokens=LLM_MAX_TOKENS,
            )
            result = resp.choices[0].message.content
            elapsed = time.time() - t0
            print(f"({elapsed:.1f}s)", end=" ", flush=True)

            with open(cache_file, 'w') as f:
                json.dump(result, f, ensure_ascii=False)

        # Parse and report
        cfds = parse_cfd_response(result, valid_columns)
        n_cfd = len(cfds)
        has_consistency = any(c.get('type') == 'consistency' for c in cfds)
        print(f"{n_cfd} CFDs, consistency={'Yes' if has_consistency else 'No'}")

        run_results.append({
            'run_id': run_id,
            'n_cfd': n_cfd,
            'has_consistency': has_consistency,
            'response_hash': hashlib.md5(result.encode()).hexdigest(),
            'response': result,
        })

    # ============================================================
    # DETERMINISM CHECK
    # ============================================================
    print("\n" + "=" * 70)
    print("DETERMINISM CHECK")
    print("=" * 70)

    hashes = [r['response_hash'] for r in run_results]
    unique_hashes = set(hashes)
    print(f"Unique responses: {len(unique_hashes)} / {N_RUNS}")

    if len(unique_hashes) == 1:
        print("✓ ALL 15 RUNS ARE IDENTICAL — temp=0.0 is fully deterministic")
        print("  → HSCV is unnecessary (no variance to vote on)")
        print("  → Single API call suffices for deployment")
    else:
        print(f"⚠ {len(unique_hashes)} distinct responses found")
        # Group by hash
        from collections import Counter
        hash_counts = Counter(hashes)
        for h, count in hash_counts.most_common():
            sample_run = next(r for r in run_results if r['response_hash'] == h)
            print(f"  Hash {h[:8]}: {count} runs, {sample_run['n_cfd']} CFDs, consistency={'Yes' if sample_run['has_consistency'] else 'No'}")

    # Summary stats
    n_cfds = [r['n_cfd'] for r in run_results]
    n_consistency = sum(1 for r in run_results if r['has_consistency'])
    print(f"\nCFD count: min={min(n_cfds)}, max={max(n_cfds)}, mean={sum(n_cfds)/len(n_cfds):.1f}")
    print(f"Runs with consistency: {n_consistency}/{N_RUNS}")

    print(f"\n✓ {N_RUNS} cot_fewshot runs saved to {CACHE_DIR}")
    print("  Ready for unified_experiment.py")

if __name__ == "__main__":
    main()
