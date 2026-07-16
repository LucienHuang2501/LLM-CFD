#!/usr/bin/env python3
"""
Generate 4 additional LLM runs (run_id=10..13) to reach 15 total.
Uses the same few_shot prompt as supplementary_experiments.py.
"""
import json, os, sys, hashlib, time, warnings
import pandas as pd
from openai import OpenAI
warnings.filterwarnings('ignore')

# Import shared functions from supplementary_experiments
sys.path.insert(0, "/sessions/69e790d3616697199023cd0a/workspace/mqw7dr6nxdtvfujn/experiments/src")
from supplementary_experiments import (
    DATA_PATH, CACHE_PATH, DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS,
    load_data, build_schema, build_prompt, call_llm,
)

def main():
    print("=" * 60)
    print("GENERATING 4 ADDITIONAL LLM RUNS (run_id=10..13)")
    print("=" * 60)

    df = load_data()
    schema = build_schema(df)
    system_msg, user_msg = build_prompt(schema, strategy="few_shot")

    # Check existing cache files
    existing = sorted([f for f in os.listdir(CACHE_PATH) if f.endswith('.json')])
    print(f"Existing cache files: {len(existing)}")

    # Generate 4 new runs
    for run_id in range(10, 14):
        cache_key = hashlib.md5(f"{system_msg}{user_msg}{run_id}".encode()).hexdigest()
        cache_file = os.path.join(CACHE_PATH, f"{cache_key}.json")

        if os.path.exists(cache_file):
            print(f"  Run {run_id}: Already cached, skipping")
            continue

        print(f"  Run {run_id}: Calling DeepSeek API...", end=" ", flush=True)
        t0 = time.time()

        client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
        resp = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
        )
        result = resp.choices[0].message.content
        elapsed = time.time() - t0

        with open(cache_file, 'w') as f:
            json.dump(result, f, ensure_ascii=False)

        # Quick parse check
        from supplementary_experiments import parse_cfds
        cfds = parse_cfds(result)
        print(f"Done ({elapsed:.1f}s) - {len(cfds)} CFDs")

    # Verify total
    all_files = sorted([f for f in os.listdir(CACHE_PATH) if f.endswith('.json')])
    print(f"\nTotal cache files now: {len(all_files)}")
    print("✓ Ready for 15-run experiments")

if __name__ == "__main__":
    main()
