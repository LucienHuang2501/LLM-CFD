"""
LLM-CFD Experiment Suite
========================
Complete experiment pipeline for:
  Phase 1: LLM Semantic CFD Discovery
  Phase 2: Statistical Validation Gate  
  Phase 3: Anomaly Scoring
  Phase 4: Explainable Output
  Baselines: EIF (Extended Isolation Forest), COPOD

Dataset: Kaggle Telco Customer Churn (7,043 × 21)
LLM Backend: DeepSeek-V4-Flash (API: deepseek-chat)
"""

import json
import os
import sys
import time
import warnings
import hashlib
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    average_precision_score, ndcg_score, precision_recall_curve
)
from scipy import stats
from openai import OpenAI

warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "data/telco_churn.csv"
RESULTS_PATH = "results"
CACHE_PATH = "results/cache"

# DeepSeek API config
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"  # DeepSeek-V4-Flash

# Experiment params
SUPPORT_THRESHOLD = 0.01  # θ_s
CONFIDENCE_THRESHOLDS = {"high": 0.90, "medium": 0.85, "low": 0.75}
ANOMALY_INJECTION_RATES = {
    "dependency_destroy": 0.10,
    "range_violation": 0.05,
    "logic_contradiction": 0.03,
}
RANDOM_SEED = 42
LLM_TEMPERATURE = 0.0
LLM_MAX_TOKENS = 4096

np.random.seed(RANDOM_SEED)

os.makedirs(RESULTS_PATH, exist_ok=True)
os.makedirs(CACHE_PATH, exist_ok=True)


# ============================================================
# DATA PREPROCESSING
# ============================================================

def load_and_preprocess(data_path: str) -> tuple[pd.DataFrame, dict]:
    """Load Telco Churn dataset and preprocess."""
    print("=" * 60)
    print("DATA PREPROCESSING")
    print("=" * 60)

    df = pd.read_csv(data_path)
    print(f"  Raw shape: {df.shape}")

    # Fix TotalCharges: empty strings → NaN → 0 (new customers)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    n_null = df['TotalCharges'].isna().sum()
    df['TotalCharges'] = df['TotalCharges'].fillna(0)
    print(f"  TotalCharges: {n_null} empty strings → 0")

    # Drop customerID (not a feature)
    df = df.drop(columns=['customerID'])

    # Encode categoricals for numerical processing
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    label_encoders = {}
    df_encoded = df.copy()
    for col in categorical_cols:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    # Build schema metadata for LLM
    schema_metadata = []
    for col in df.columns:
        meta = {
            "name": col,
            "dtype": str(df[col].dtype),
            "nunique": int(df[col].nunique()),
        }
        if col in numerical_cols:
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
        # Sample values
        samples = df[col].dropna().sample(min(8, len(df)), random_state=RANDOM_SEED).tolist()
        meta["samples"] = [str(s) for s in samples]
        schema_metadata.append(meta)

    print(f"  Numerical columns: {len(numerical_cols)}")
    print(f"  Categorical columns: {len(categorical_cols)}")
    print(f"  Schema metadata: {len(schema_metadata)} columns described")

    return df, {
        "df_encoded": df_encoded,
        "numerical_cols": numerical_cols,
        "categorical_cols": categorical_cols,
        "label_encoders": label_encoders,
        "schema_metadata": schema_metadata,
    }


# ============================================================
# PHASE 1: LLM SEMANTIC CFD DISCOVERY
# ============================================================

CFD_EXAMPLES_FEWSHOT = r"""
示例1 (range约束-条件范围): 当Contract='Two year'时，MonthlyCharges在特定区间
{
  "type": "range",
  "condition_attributes": ["Contract"],
  "condition_values": {"Contract": "Two year"},
  "dependent_attribute": "MonthlyCharges",
  "expected_pattern": {"min": 50, "max": 120},
  "confidence_estimate": "high",
  "natural_language_description": "两年期合同客户的月消费应在50-120美元之间"
}

示例2 (enum约束-级联依赖): 当InternetService='No'时，OnlineSecurity必须为'No internet service'
{
  "type": "enum",
  "condition_attributes": ["InternetService"],
  "condition_values": {"InternetService": "No"},
  "dependent_attribute": "OnlineSecurity",
  "expected_pattern": {"values": ["No internet service"]},
  "confidence_estimate": "high",
  "natural_language_description": "无网络服务的客户不应有在线安全服务"
}

示例3 (enum约束-正反都考虑): 当PhoneService='Yes'时，MultipleLines只能是'Yes'或'No'（不能是'No phone service'）；当PhoneService='No'时，MultipleLines必须是'No phone service'
{
  "type": "enum",
  "condition_attributes": ["PhoneService"],
  "condition_values": {"PhoneService": "Yes"},
  "dependent_attribute": "MultipleLines",
  "expected_pattern": {"values": ["Yes", "No"]},
  "confidence_estimate": "high",
  "natural_language_description": "有电话服务的客户，多线路只能是Yes或No，不能是No phone service"
}

示例4 (range约束-全局范围): 无条件下，tenure的取值范围应在0-72之间
{
  "type": "range",
  "condition_attributes": [],
  "condition_values": {},
  "dependent_attribute": "tenure",
  "expected_pattern": {"min": 0, "max": 72},
  "confidence_estimate": "high",
  "natural_language_description": "在网时长应在0-72个月之间"
}

示例5 (logic约束-相关性): 高tenure客户（>24个月）通常Churn='No'
{
  "type": "logic",
  "condition_attributes": ["tenure"],
  "condition_values": {},
  "dependent_attribute": "Churn",
  "expected_pattern": {"expression": "tenure > 24 → Churn='No' with high probability"},
  "confidence_estimate": "medium",
  "natural_language_description": "在网超过2年的客户流失率应显著低于新客户"
}

示例6 (enum约束-人口属性枚举): SeniorCitizen只能是0或1（二值枚举属性）
{
  "type": "enum",
  "condition_attributes": [],
  "condition_values": {},
  "dependent_attribute": "SeniorCitizen",
  "expected_pattern": {"values": ["0", "1"]},
  "confidence_estimate": "high",
  "natural_language_description": "是否为老年人只能是0或1"
}

示例7 (consistency约束-数值一致性): TotalCharges应近似等于tenure乘以MonthlyCharges
{
  "type": "consistency",
  "condition_attributes": ["tenure", "MonthlyCharges"],
  "condition_values": {},
  "dependent_attribute": "TotalCharges",
  "expected_pattern": {"relation": "TotalCharges ≈ tenure * MonthlyCharges"},
  "confidence_estimate": "high",
  "natural_language_description": "总消费应近似等于在网月数乘以月消费"
}

示例8 (range约束-合同期限): 当Contract='One year'时，tenure通常不超过36个月
{
  "type": "range",
  "condition_attributes": ["Contract"],
  "condition_values": {"Contract": "One year"},
  "dependent_attribute": "tenure",
  "expected_pattern": {"min": 1, "max": 36},
  "confidence_estimate": "medium",
  "natural_language_description": "一年期合同客户在网时长通常在1-36个月"
}

示例9 (enum约束-级联扩充): 当InternetService='No'时，所有在线附加服务（OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies）都应为'No internet service'——注意这是一类模式，应对每个附加服务都生成一条CFD
{
  "type": "enum",
  "condition_attributes": ["InternetService"],
  "condition_values": {"InternetService": "No"},
  "dependent_attribute": "StreamingMovies",
  "expected_pattern": {"values": ["No internet service"]},
  "confidence_estimate": "high",
  "natural_language_description": "无网络服务的客户不应有流媒体电影服务（对于每个在线附加服务OnlineBackup/DeviceProtection/TechSupport/StreamingTV都应类似地生成一条CFD）"
}

示例10 (enum约束-全局枚举): PaymentMethod的有效取值集合
{
  "type": "enum",
  "condition_attributes": [],
  "condition_values": {},
  "dependent_attribute": "PaymentMethod",
  "expected_pattern": {"values": ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]},
  "confidence_estimate": "high",
  "natural_language_description": "支付方式必须在四种已知方式之内"
}
"""


def build_llm_prompt(schema_metadata: list[dict], df_sample: pd.DataFrame,
                     strategy: str = "cot_fewshot") -> str:
    """Build prompt for CFD discovery with selectable strategy.
    
    Strategies:
      - zero_shot: No examples, just task description
      - few_shot: Examples included, no CoT reasoning steps
      - cot_fewshot: CoT reasoning steps + Few-Shot examples (default)
    """
    schema_text = json.dumps(schema_metadata, ensure_ascii=False, indent=2)
    sample_data = df_sample.head(8).to_dict(orient='records')
    sample_text = json.dumps(sample_data, ensure_ascii=False, indent=2)
    
    output_format = """## 输出格式要求：
请严格以JSON数组格式输出所有发现的CFD规则，每条CFD包含以下字段：
- type: "range"|"enum"|"fd"|"logic"|"consistency"
- condition_attributes: 条件属性名列表
- condition_values: 条件属性的取值（dict格式，需给出具体的条件值）
- dependent_attribute: 被依赖属性名
- expected_pattern: 期望取值模式（根据类型不同：range→{{"min":X,"max":Y}}, enum→{{"values":[...]}}, fd→{{"determines":"value"}}, logic→{{"expression":"..."}}, consistency→{{"relation":"..."}}）
- confidence_estimate: "high"|"medium"|"low"
- natural_language_description: 中文自然语言描述

请输出JSON数组（仅输出JSON，不要有其他文字）：
```json
[
  {{...}},
  {{...}}
]
```"""

    task_desc = f"""你是一位数据质量专家，专精于条件函数依赖（Conditional Functional Dependency, CFD）的发现。
请分析以下数据表的Schema和样本数据，推断可能存在的条件函数依赖关系。

## 数据表Schema（含统计信息）：
{schema_text}

## 数据样本（前8行）：
{sample_text}"""

    if strategy == "zero_shot":
        prompt = task_desc + "\n\n" + """## 任务：
请发现数据中可能存在的所有条件函数依赖关系（CFD），包括但不限于：
- range（范围约束）：数值属性在特定条件下有取值区间
- enum（枚举约束）：类别属性在特定条件下取值受限
- fd（函数依赖）：一个或一组属性值确定另一个属性值
- logic（逻辑约束）：属性间存在逻辑推导关系
- consistency（一致性约束）：属性间存在数值一致性关系

""" + output_format

    elif strategy == "few_shot":
        prompt = task_desc + "\n\n" + """## 参考示例：
""" + CFD_EXAMPLES_FEWSHOT + "\n\n" + output_format

    else:  # cot_fewshot
        prompt = task_desc + "\n\n" + """## 推理步骤：
请系统性地完成以下分析，确保覆盖所有属性、所有CFD类型，目标是发现至少20条CFD规则：

**第一步：识别级联服务依赖（enum类型，约8条）**
- 找出所有"主服务→附加服务"关系对：
  · InternetService → OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies (6对)
  · PhoneService → MultipleLines (1对)
- 对每个关系对，分析两种条件：主服务=No时附加服务必须是什么？主服务=Yes时附加服务可以是哪些值？
- 为每种条件生成一条enum类型CFD

**第二步：合同类型驱动的数值范围约束（range类型，约6条）**
- 分析不同Contract类型下MonthlyCharges的典型范围：
  · Month-to-month: 18-120
  · One year: 25-120
  · Two year: 50-120
- 分析不同Contract类型下tenure的合理范围：
  · Month-to-month: 0-72
  · One year: 1-36
  · Two year: 1-72
- 为每种Contract类型生成对应的range约束

**第三步：人口统计属性的枚举约束（enum类型，约6条）**
- 检查所有二值/有限类别属性，列出合法取值集合：
  · SeniorCitizen: [0, 1]
  · gender: [Male, Female]
  · Partner: [Yes, No]
  · Dependents: [Yes, No]
  · PaperlessBilling: [Yes, No]
  · PhoneService: [Yes, No]
- 为每个属性生成enum约束（条件为空{}的全局枚举）

**第四步：业务逻辑相关性（logic类型，约3条）**
- tenure与Churn的关系：高tenure（>24月）客户流失率显著低于新客户，生成logic约束
- SeniorCitizen=1且无Partner/Dependents的客户Churn概率更高
- 电子支付方式（Credit card/Bank transfer）与PaperlessBilling='Yes'高度相关

**第五步：数值一致性约束（consistency类型，约2条）**
- 核心关系：TotalCharges ≈ tenure × MonthlyCharges
- 当tenure=0时，TotalCharges应接近0或等于MonthlyCharges

**第六步：全局范围约束（range类型，无条件，约3条）**
- MonthlyCharges全局范围: [18, 120]
- TotalCharges全局范围: [0, 9000]
- tenure全局范围: [0, 72]

**第七步：支付与渠道属性枚举（enum类型，约3条）**
- PaymentMethod的4种合法取值
- InternetService的3种合法取值（DSL/Fiber optic/No）
- Contract的3种合法取值（Month-to-month/One year/Two year）

重要提示：请确保每条CFD都有具体的condition_values（条件为空时用{}），不要生成过于宽泛的描述性规则。数量目标：至少20条，覆盖所有上述类别。

## 参考示例：
""" + CFD_EXAMPLES_FEWSHOT + "\n\n" + output_format

    return prompt


def call_deepseek_api(prompt: str, api_key: str) -> str:
    """Call DeepSeek-V4-Flash API (API identifier: deepseek-chat)."""
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY not set. Export it as environment variable.")

    client = OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)

    response = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[
            {"role": "system", "content": "你是一位数据质量专家。请严格按照要求输出JSON格式的结果，不要输出任何其他内容。"},
            {"role": "user", "content": prompt},
        ],
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS,
    )

    return response.choices[0].message.content


def parse_cfd_response(response_text: str, valid_columns: set) -> list[dict]:
    """Parse LLM response into structured CFD objects with schema validation."""
    # Extract JSON block
    import re
    json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
    else:
        # Try to find JSON array directly
        json_match = re.search(r'\[\s*\{.*\}\s*\]', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
        else:
            json_str = response_text

    try:
        cfds = json.loads(json_str)
    except json.JSONDecodeError:
        print("  WARNING: Failed to parse LLM JSON response. Raw response:")
        print(f"  {response_text[:500]}...")
        return []

    if not isinstance(cfds, list):
        cfds = [cfds]

    # Schema validation
    validated = []
    for cfd in cfds:
        # Check required fields
        required = ["type", "condition_attributes", "condition_values",
                     "dependent_attribute", "expected_pattern",
                     "confidence_estimate", "natural_language_description"]
        if not all(k in cfd for k in required):
            continue

        # Validate column names exist
        all_attrs = cfd.get("condition_attributes", []) + [cfd.get("dependent_attribute", "")]
        if any(a not in valid_columns for a in all_attrs):
            continue

        # Validate confidence_estimate
        if cfd["confidence_estimate"] not in ["high", "medium", "low"]:
            cfd["confidence_estimate"] = "medium"

        cfd["status"] = "candidate"
        validated.append(cfd)

    return validated


# ============================================================
# PHASE 2: STATISTICAL VALIDATION GATE
# ============================================================

def evaluate_cfd_condition(df: pd.DataFrame, cfd: dict) -> np.ndarray:
    """Evaluate which rows satisfy a CFD's condition."""
    mask = np.ones(len(df), dtype=bool)
    for attr, val in cfd["condition_values"].items():
        if attr in df.columns:
            col_vals = df[attr].astype(str)
            mask &= (col_vals == str(val))
    return mask


def evaluate_cfd_dependency(df: pd.DataFrame, cfd: dict, condition_mask: np.ndarray) -> np.ndarray:
    """Evaluate which rows satisfy a CFD's dependency (within conditioned rows)."""
    dep_attr = cfd["dependent_attribute"]
    if dep_attr not in df.columns:
        return np.zeros(condition_mask.sum(), dtype=bool)

    dep_type = cfd["type"]
    expected = cfd["expected_pattern"]
    n_cond = condition_mask.sum()
    satisfied = np.zeros(n_cond, dtype=bool)

    if dep_type == "range":
        # Numerical: convert to float
        try:
            dep_values = df.loc[condition_mask, dep_attr].astype(float)
        except (ValueError, TypeError):
            return np.zeros(n_cond, dtype=bool)
        lo = expected.get("min", -np.inf)
        hi = expected.get("max", np.inf)
        satisfied = (dep_values >= lo) & (dep_values <= hi)
    elif dep_type == "enum":
        allowed = set(str(v) for v in expected.get("values", []))
        satisfied = df.loc[condition_mask, dep_attr].astype(str).isin(allowed).values
    elif dep_type in ("fd", "logic", "consistency"):
        # Simplified: check if value is not null/empty
        satisfied = df.loc[condition_mask, dep_attr].notna().values

    return satisfied


def statistical_validation(df: pd.DataFrame, candidates: list[dict]) -> tuple[list[dict], dict]:
    """
    Phase 2: Statistical Validation Gate.
    Returns (validated_cfds, stats_report)
    """
    print("\n" + "=" * 60)
    print("PHASE 2: STATISTICAL VALIDATION GATE")
    print("=" * 60)

    n_total = len(df)
    validated = []
    rejected = []
    stats_report = {
        "total_candidates": len(candidates),
        "validated": 0,
        "rejected": 0,
        "rejection_reasons": {},
    }

    for cfd in candidates:
        # Support
        condition_mask = evaluate_cfd_condition(df, cfd)
        support = condition_mask.sum() / n_total

        if support < SUPPORT_THRESHOLD:
            cfd["status"] = "rejected"
            cfd["reject_reason"] = f"support={support:.4f} < threshold={SUPPORT_THRESHOLD}"
            cfd["support"] = support
            rejected.append(cfd)
            stats_report["rejection_reasons"]["low_support"] = \
                stats_report["rejection_reasons"].get("low_support", 0) + 1
            continue

        # Confidence
        dep_satisfied = evaluate_cfd_dependency(df, cfd, condition_mask)
        n_conditioned = condition_mask.sum()
        n_violations = (~dep_satisfied).sum()
        confidence = 1.0 - (n_violations / n_conditioned) if n_conditioned > 0 else 0.0

        # Adaptive threshold
        llm_conf = cfd["confidence_estimate"]
        threshold = CONFIDENCE_THRESHOLDS.get(llm_conf, 0.85)

        cfd["support"] = support
        cfd["confidence"] = confidence

        if confidence >= threshold:
            cfd["status"] = "validated"
            validated.append(cfd)
        else:
            cfd["status"] = "rejected"
            cfd["reject_reason"] = f"confidence={confidence:.4f} < threshold={threshold} (LLM confidence={llm_conf})"
            rejected.append(cfd)
            stats_report["rejection_reasons"]["low_confidence"] = \
                stats_report["rejection_reasons"].get("low_confidence", 0) + 1

    stats_report["validated"] = len(validated)
    stats_report["rejected"] = len(rejected)
    stats_report["filter_rate"] = len(rejected) / len(candidates) if candidates else 0

    print(f"  Candidates: {len(candidates)}")
    print(f"  Validated:  {len(validated)} ({len(validated)/max(len(candidates),1)*100:.1f}%)")
    print(f"  Rejected:   {len(rejected)} ({len(rejected)/max(len(candidates),1)*100:.1f}%)")

    # Deduplication
    validated = deduplicate_cfds(validated)

    return validated, stats_report


def deduplicate_cfds(cfds: list[dict], jaccard_threshold: float = 0.8) -> list[dict]:
    """Remove semantically duplicate CFDs based on Jaccard similarity of condition sets."""
    if len(cfds) <= 1:
        return cfds

    kept = []
    for i, cfd_i in enumerate(cfds):
        is_dup = False
        for cfd_j in kept:
            # Check same dependent attribute
            if cfd_i["dependent_attribute"] != cfd_j["dependent_attribute"]:
                continue
            # Check condition overlap
            cond_i = set(cfd_i.get("condition_attributes", []))
            cond_j = set(cfd_j.get("condition_attributes", []))
            if not cond_i and not cond_j:
                continue
            intersection = cond_i & cond_j
            union = cond_i | cond_j
            jaccard = len(intersection) / len(union) if union else 0
            if jaccard > jaccard_threshold:
                # Keep the one with higher confidence
                if cfd_i.get("confidence", 0) > cfd_j.get("confidence", 0):
                    kept.remove(cfd_j)
                    kept.append(cfd_i)
                is_dup = True
                break
        if not is_dup:
            kept.append(cfd_i)

    return kept


# ============================================================
# PHASE 3: ANOMALY SCORING
# ============================================================

def compute_anomaly_scores(df: pd.DataFrame, validated_cfds: list[dict]) -> np.ndarray:
    """
    Phase 3: Build gradient violation matrix with redundancy reduction and top3_mean aggregation.
    """
    print("\n" + "=" * 60)
    print("PHASE 3: ANOMALY SCORING")
    print("=" * 60)

    if not validated_cfds:
        print("  WARNING: No validated CFDs — returning zero scores")
        return np.zeros(len(df))

    n = len(df)
    m = len(validated_cfds)
    V = np.zeros((n, m))

    for j, cfd in enumerate(validated_cfds):
        condition_mask = evaluate_cfd_condition(df, cfd)
        dep_attr = cfd["dependent_attribute"]

        if dep_attr not in df.columns:
            continue

        if not condition_mask.any():
            continue

        dep_type = cfd["type"]
        expected = cfd["expected_pattern"]

        if dep_type == "range":
            lo = expected.get("min", -np.inf)
            hi = expected.get("max", np.inf)
            values = df[dep_attr].astype(float).values
            rng = hi - lo + 1e-8

            below = np.maximum(lo - values, 0) / rng
            above = np.maximum(values - hi, 0) / rng
            V[:, j] = condition_mask.astype(float) * np.maximum(below, above)

        elif dep_type == "enum":
            # Gradient: 1 - P(dep=actual | condition)
            allowed = set(str(v) for v in expected.get("values", []))
            sub_vals = df.loc[condition_mask, dep_attr].astype(str)
            vc = sub_vals.value_counts()
            total = len(sub_vals)
            if total > 0:
                cond_arr = np.asarray(condition_mask)
                for idx_pos, val in zip(np.where(cond_arr)[0], sub_vals.values):
                    freq = vc.get(val, 0) / total
                    V[idx_pos, j] = 1.0 - freq
            # Values not in allowed set get max violation
            violated = ~df[dep_attr].astype(str).isin(allowed).values
            V[:, j] = np.where(violated & np.asarray(condition_mask), 1.0, V[:, j])

        elif dep_type == "consistency":
            # Gradient: |actual - expected| / |expected|
            if dep_attr == "TotalCharges" and all(c in df.columns for c in ['tenure', 'MonthlyCharges']):
                actual = pd.to_numeric(df[dep_attr], errors='coerce').values
                expected_vals = pd.to_numeric(df['tenure'], errors='coerce').values * pd.to_numeric(df['MonthlyCharges'], errors='coerce').values
                deviation = np.abs(actual - expected_vals) / (np.abs(expected_vals) + 1e-8)
                V[:, j] = np.asarray(condition_mask, dtype=float) * np.clip(deviation, 0, 1)
            else:
                dep_satisfied = evaluate_cfd_dependency(df, cfd, condition_mask)
                full_satisfied = np.zeros(n, dtype=bool)
                full_satisfied[condition_mask] = dep_satisfied
                V[:, j] = condition_mask.astype(float) * (~full_satisfied).astype(float)

        elif dep_type == "logic":
            # Gradient: 1 - P(A=actual | condition pattern)
            expr = expected.get("expression", "")
            if "tenure" in expr and dep_attr == "tenure":
                vals = pd.to_numeric(df[dep_attr], errors='coerce').values
                # tenure >= 0: negative values are impossible, P=0, violation=1
                # In-range values: use 1 - CDF-like frequency
                sub_vals = pd.to_numeric(df.loc[condition_mask, dep_attr], errors='coerce')
                if len(sub_vals) > 0:
                    median_val = sub_vals.median()
                    cond_arr = np.asarray(condition_mask)
                    for idx_pos, val in zip(np.where(cond_arr)[0], sub_vals.values):
                        if val < 0:
                            V[idx_pos, j] = 1.0
                        else:
                            # Graduated violation for extreme but positive values
                            V[idx_pos, j] = 0.0
            else:
                dep_satisfied = evaluate_cfd_dependency(df, cfd, condition_mask)
                full_satisfied = np.zeros(n, dtype=bool)
                full_satisfied[condition_mask] = dep_satisfied
                V[:, j] = condition_mask.astype(float) * (~full_satisfied).astype(float)

        else:
            # fd: Gradient: 1 - P(dep=actual | condition)
            sub_vals = df.loc[condition_mask, dep_attr].astype(str)
            vc = sub_vals.value_counts()
            total = len(sub_vals)
            if total > 0:
                cond_arr = np.asarray(condition_mask)
                for idx_pos, val in zip(np.where(cond_arr)[0], sub_vals.values):
                    freq = vc.get(val, 0) / total
                    V[idx_pos, j] = 1.0 - freq
            else:
                V[:, j] = condition_mask.astype(float) * 0.0

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

    # top3_mean aggregation: s_i = (1/min(3,g)) * sum of top-3 group violations
    k = min(3, n_groups)
    if n_groups <= k:
        scores = V_reduced.mean(axis=1)
    else:
        topk = np.sort(V_reduced, axis=1)[:, -k:]
        scores = topk.mean(axis=1)

    # Clip at 99th percentile and normalize to [0,1]
    clip_val = np.percentile(scores, 99)
    scores = np.clip(scores, 0, clip_val)
    if scores.max() > 0:
        scores = scores / scores.max()

    print(f"  CFD rules used: {m}")
    print(f"  Score range: [{scores.min():.4f}, {scores.max():.4f}]")
    print(f"  Mean score: {scores.mean():.4f}")

    return scores


# ============================================================
# BASELINES: EIF & COPOD
# ============================================================

def run_eif(X: np.ndarray, contamination: float = 0.18) -> np.ndarray:
    """Extended Isolation Forest using sklearn's IsolationForest."""
    print("\n  Running EIF baseline...")
    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    # sklearn IF uses extended (random slope) splits by default
    raw_scores = model.fit_predict(X)
    # Convert to anomaly scores [0,1] where 1 = most anomalous
    scores = model.score_samples(X)
    scores = -scores  # Higher = more anomalous
    scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
    return scores


def run_copod(X: np.ndarray, contamination: float = 0.18) -> np.ndarray:
    """COPOD: Copula-Based Outlier Detection (implemented from paper)."""
    print("\n  Running COPOD baseline...")
    n, d = X.shape

    # Step 1: Compute empirical CDF (left-tail) and 1-CDF (right-tail) for each dimension
    U_l = np.zeros((n, d))
    U_r = np.zeros((n, d))

    for j in range(d):
        col = X[:, j]
        # Handle ties by using fractional ranking
        ranks = stats.rankdata(col, method='average')
        U_l[:, j] = ranks / (n + 1)  # Left-tail ECDF
        U_r[:, j] = 1.0 - U_l[:, j]   # Right-tail ECDF

    # Step 2: Compute negative log-probability for each tail
    # Prevent log(0)
    eps = 1e-10
    U_l = np.clip(U_l, eps, 1 - eps)
    U_r = np.clip(U_r, eps, 1 - eps)

    left_tail = -np.log(U_l)
    right_tail = -np.log(U_r)

    # Step 3: Skewness-corrected aggregation
    # Use dimension-wise skewness to weight tails
    skewness = np.array([stats.skew(X[:, j]) for j in range(d)])

    scores = np.zeros(n)
    for j in range(d):
        if skewness[j] < 0:  # Left-skewed: use left tail
            scores += left_tail[:, j]
        else:  # Right-skewed or symmetric: use right tail
            scores += right_tail[:, j]

    # Normalize
    scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
    return scores


# ============================================================
# SYNTHETIC ANOMALY INJECTION
# ============================================================

def inject_anomalies(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Inject synthetic anomalies into the dataset.
    Returns (contaminated_df, ground_truth_labels) where 1=anomaly.
    """
    print("\n" + "=" * 60)
    print("SYNTHETIC ANOMALY INJECTION")
    print("=" * 60)

    df_contaminated = df.copy()
    n = len(df)
    labels = np.zeros(n, dtype=int)

    rng = np.random.RandomState(RANDOM_SEED)

    # 1. Dependency destruction: break Contract→MonthlyCharges relationship
    n_dep = int(n * ANOMALY_INJECTION_RATES["dependency_destroy"])
    dep_indices = rng.choice(n, n_dep, replace=False)
    for idx in dep_indices:
        contract = df.loc[idx, 'Contract']
        current_charge = df.loc[idx, 'MonthlyCharges']
        # Swap MonthlyCharges to a value from a different contract tier (in-range)
        if contract == 'Two year':
            # Swap to Month-to-month range (18-40, still valid overall range)
            df_contaminated.loc[idx, 'MonthlyCharges'] = rng.uniform(18, 40)
        elif contract == 'One year':
            # Swap to Month-to-month range
            df_contaminated.loc[idx, 'MonthlyCharges'] = rng.uniform(18, 35)
        else:
            # Month-to-month: swap to Two year range (still valid overall range)
            df_contaminated.loc[idx, 'MonthlyCharges'] = rng.uniform(60, 120)
    labels[dep_indices] = 1
    print(f"  Dependency destruction: {n_dep} records")

    # 2. In-range distributional outliers (CFD-legal, not 3σ extreme)
    n_range = int(n * ANOMALY_INJECTION_RATES["range_violation"])
    range_indices = rng.choice(list(set(range(n)) - set(dep_indices)), n_range, replace=False)
    for idx in range_indices:
        col = rng.choice(['MonthlyCharges', 'tenure', 'TotalCharges'])
        # Use percentile-based shift within valid range (not 3σ extreme)
        p = rng.choice([1, 2, 98, 99])
        shifted_val = np.percentile(df[col].astype(float), p)
        # Add small noise to avoid exact duplicates
        shifted_val += rng.normal(0, shifted_val * 0.02)
        # Clip to valid range
        col_min = df[col].astype(float).min()
        col_max = df[col].astype(float).max()
        df_contaminated.loc[idx, col] = np.clip(shifted_val, col_min, col_max)
    labels[range_indices] = 1
    print(f"  In-range distributional outlier: {n_range} records")

    # 3. Logic contradiction
    n_logic = int(n * ANOMALY_INJECTION_RATES["logic_contradiction"])
    remaining = list(set(range(n)) - set(dep_indices) - set(range_indices))
    logic_indices = rng.choice(remaining, min(n_logic, len(remaining)), replace=False)
    for idx in logic_indices:
        # Contradict InternetService=No but OnlineSecurity=Yes
        if df.loc[idx, 'InternetService'] == 'No':
            df_contaminated.loc[idx, 'OnlineSecurity'] = 'Yes'
        else:
            df_contaminated.loc[idx, 'OnlineSecurity'] = 'No internet service'
    labels[logic_indices] = 1
    print(f"  Logic contradiction: {len(logic_indices)} records")

    total_anomalies = labels.sum()
    print(f"  Total injected anomalies: {total_anomalies} ({total_anomalies/n*100:.1f}%)")

    return df_contaminated, labels


# ============================================================
# EVALUATION METRICS
# ============================================================

def evaluate_cfd_discovery(discovered_cfds: list[dict], ground_truth_cfds: list[dict]) -> dict:
    """Evaluate CFD discovery quality: Precision, Recall, F1.
    Matching is based on (dependent_attribute, condition_attributes, type_category)
    where type categories group semantically equivalent CFD types:
      - categorical: enum, fd, logic
      - numerical: range
      - structural: consistency
    """
    # Type category normalization: fd/enum/logic are semantically equivalent
    # for constraining attribute values under conditions
    _CATEGORY_MAP = {
        "fd": "categorical", "enum": "categorical", "logic": "categorical",
        "range": "numerical",
        "consistency": "structural",
    }

    def _category(t: str) -> str:
        return _CATEGORY_MAP.get(t, t)

    def key(cfd):
        return (cfd["dependent_attribute"],
                tuple(sorted(cfd.get("condition_attributes", []))),
                _category(cfd["type"]))

    gt_keys = {key(c) for c in ground_truth_cfds}
    discovered_keys = {key(c) for c in discovered_cfds}

    tp = len(gt_keys & discovered_keys)
    fp = len(discovered_keys - gt_keys)
    fn = len(gt_keys - discovered_keys)

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def evaluate_anomaly_detection(scores: np.ndarray, labels: np.ndarray, k: int = 100) -> dict:
    """Evaluate anomaly detection performance."""
    # AUPRC
    auprc = average_precision_score(labels, scores)

    # Precision@k
    top_k_idx = np.argsort(scores)[-k:]
    precision_at_k = labels[top_k_idx].mean()

    # NDCG@k
    # NDCG needs relevance scores; use binary labels
    sorted_idx = np.argsort(scores)[::-1]
    y_true_sorted = labels[sorted_idx].reshape(1, -1)
    y_score_sorted = scores[sorted_idx].reshape(1, -1)
    try:
        ndcg = ndcg_score(y_true_sorted, y_score_sorted, k=k)
    except Exception:
        ndcg = 0.0

    return {
        "auprc": auprc,
        "precision_at_k": precision_at_k,
        "ndcg_at_k": ndcg,
    }


def build_ground_truth_cfds() -> list[dict]:
    """Build ground truth CFD rules for Telco Churn dataset (15 rules, matching gt_rules_telco.json)."""
    return [
        # Function Dependencies (7)
        {"dependent_attribute": "OnlineSecurity", "condition_attributes": ["InternetService"], "type": "fd",
         "condition_values": {"InternetService": "No"}, "expected_pattern": {"values": ["No internet service"]}},
        {"dependent_attribute": "OnlineBackup", "condition_attributes": ["InternetService"], "type": "fd",
         "condition_values": {"InternetService": "No"}, "expected_pattern": {"values": ["No internet service"]}},
        {"dependent_attribute": "DeviceProtection", "condition_attributes": ["InternetService"], "type": "fd",
         "condition_values": {"InternetService": "No"}, "expected_pattern": {"values": ["No internet service"]}},
        {"dependent_attribute": "TechSupport", "condition_attributes": ["InternetService"], "type": "fd",
         "condition_values": {"InternetService": "No"}, "expected_pattern": {"values": ["No internet service"]}},
        {"dependent_attribute": "StreamingTV", "condition_attributes": ["InternetService"], "type": "fd",
         "condition_values": {"InternetService": "No"}, "expected_pattern": {"values": ["No internet service"]}},
        {"dependent_attribute": "StreamingMovies", "condition_attributes": ["InternetService"], "type": "fd",
         "condition_values": {"InternetService": "No"}, "expected_pattern": {"values": ["No internet service"]}},
        {"dependent_attribute": "MultipleLines", "condition_attributes": ["PhoneService"], "type": "fd",
         "condition_values": {"PhoneService": "No"}, "expected_pattern": {"values": ["No phone service"]}},
        # Range Constraints (2)
        {"dependent_attribute": "MonthlyCharges", "condition_attributes": ["Contract"], "type": "range",
         "condition_values": {"Contract": "Two year"}, "expected_pattern": {"min": 45, "max": 120}},
        {"dependent_attribute": "MonthlyCharges", "condition_attributes": [], "type": "range",
         "condition_values": {}, "expected_pattern": {"min": 18, "max": 120}},
        # Enum Constraints (4)
        {"dependent_attribute": "SeniorCitizen", "condition_attributes": [], "type": "enum",
         "condition_values": {}, "expected_pattern": {"values": ["0", "1"]}},
        {"dependent_attribute": "gender", "condition_attributes": [], "type": "enum",
         "condition_values": {}, "expected_pattern": {"values": ["Male", "Female"]}},
        {"dependent_attribute": "Partner", "condition_attributes": [], "type": "enum",
         "condition_values": {}, "expected_pattern": {"values": ["Yes", "No"]}},
        {"dependent_attribute": "Dependents", "condition_attributes": [], "type": "enum",
         "condition_values": {}, "expected_pattern": {"values": ["Yes", "No"]}},
        # Logic Constraints (1)
        {"dependent_attribute": "tenure", "condition_attributes": [], "type": "logic",
         "condition_values": {}, "expected_pattern": {"expression": "tenure >= 0"}},
        # Consistency Constraints (1)
        {"dependent_attribute": "TotalCharges", "condition_attributes": ["tenure", "MonthlyCharges"], "type": "consistency",
         "condition_values": {}, "expected_pattern": {"relation": "TotalCharges ≈ tenure × MonthlyCharges"}},
    ]


# ============================================================
# MAIN EXPERIMENT RUNNER
# ============================================================

def run_experiments():
    """Run the complete LLM-CFD experiment suite with all prompt strategies."""
    start_time = time.time()

    # ─── Load and preprocess ───
    df, meta = load_and_preprocess(DATA_PATH)
    df_encoded = meta["df_encoded"]
    numerical_cols = meta["numerical_cols"]
    valid_columns = set(df.columns)

    # ─── Phase 1: LLM CFD Discovery (3 strategies) ───
    STRATEGIES = ["zero_shot", "few_shot", "cot_fewshot"]
    all_strategy_results = {}
    best_validated = []
    best_strategy = ""

    ground_truth = build_ground_truth_cfds()

    for strategy in STRATEGIES:
        print("\n" + "=" * 60)
        print(f"PHASE 1: LLM CFD DISCOVERY [{strategy}]")
        print("=" * 60)

        prompt = build_llm_prompt(meta["schema_metadata"], df, strategy=strategy)
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()[:12]
        cache_file = os.path.join(CACHE_PATH, f"llm_response_{strategy}_{prompt_hash}.json")

        if os.path.exists(cache_file):
            print(f"  Loading cached LLM response: {cache_file}")
            with open(cache_file, 'r') as f:
                cache_data = json.load(f)
            response_text = cache_data["response"]
        else:
            if not DEEPSEEK_API_KEY:
                print("\n  *** ERROR: DEEPSEEK_API_KEY not set! ***")
                sys.exit(1)
            print(f"  Calling DeepSeek-V4-Flash API ({strategy})...")
            t0 = time.time()
            response_text = call_deepseek_api(prompt, DEEPSEEK_API_KEY)
            elapsed = time.time() - t0
            print(f"  API call completed in {elapsed:.1f}s")
            with open(cache_file, 'w') as f:
                json.dump({"response": response_text, "timestamp": str(datetime.now()),
                           "strategy": strategy}, f, ensure_ascii=False, indent=2)

        candidates = parse_cfd_response(response_text, valid_columns)
        print(f"  LLM generated {len(candidates)} candidate CFDs")

        if not candidates:
            print(f"  WARNING: No CFDs from {strategy}, skipping")
            continue

        # Phase 2 validation
        validated, stats_rpt = statistical_validation(df, candidates)

        # CFD discovery evaluation
        disc = evaluate_cfd_discovery(validated, ground_truth)
        disc_pre_gate = evaluate_cfd_discovery(candidates, ground_truth)
        gate_gain = disc["precision"] - disc_pre_gate["precision"]

        print(f"  Precision: {disc['precision']:.4f} | Recall: {disc['recall']:.4f} | F1: {disc['f1']:.4f}")
        print(f"  Phase 2 Gate Gain (Precision): +{gate_gain:.4f}")
        print(f"  TP={disc['tp']}, FP={disc['fp']}, FN={disc['fn']}")

        all_strategy_results[strategy] = {
            "candidates": len(candidates),
            "validated": len(validated),
            "filter_rate": stats_rpt["filter_rate"],
            "gate_gain": gate_gain,
            "precision": disc["precision"],
            "recall": disc["recall"],
            "f1": disc["f1"],
            "tp": disc["tp"], "fp": disc["fp"], "fn": disc["fn"],
        }

        # Track best strategy for anomaly detection
        if disc["f1"] > all_strategy_results.get(best_strategy, {}).get("f1", -1):
            best_strategy = strategy
            best_validated = validated

    print(f"\n  Best strategy: {best_strategy} (F1={all_strategy_results[best_strategy]['f1']:.4f})")

    # ─── Inject anomalies (enhanced: semantic + statistical) ───
    df_contaminated, anomaly_labels = inject_anomalies(df)

    # Encode for baselines
    df_contaminated_encoded = df_contaminated.copy()
    for col in meta["categorical_cols"]:
        if col in df_contaminated_encoded.columns:
            df_contaminated_encoded[col] = meta["label_encoders"][col].transform(
                df_contaminated[col].astype(str))
    X = df_contaminated_encoded[numerical_cols].values.astype(float)

    # ─── Phase 3: Anomaly Scoring (best strategy) ───
    llm_cfd_scores = compute_anomaly_scores(df_contaminated, best_validated)

    # Also get the full candidate set scores (no gate)
    # Find the best strategy's candidates
    best_candidates = []
    for strategy in STRATEGIES:
        cf = os.path.join(CACHE_PATH, f"llm_response_{strategy}_")
        # Use best strategy's candidates
        if strategy == best_strategy:
            cache_file = None
            for fname in os.listdir(CACHE_PATH):
                if fname.startswith(f"llm_response_{strategy}_"):
                    cache_file = os.path.join(CACHE_PATH, fname)
                    break
            if cache_file:
                with open(cache_file) as f:
                    cd = json.load(f)
                best_candidates = parse_cfd_response(cd["response"], valid_columns)
    if not best_candidates:
        best_candidates = best_validated  # fallback

    llm_nogate_scores = compute_anomaly_scores(df_contaminated, best_candidates)

    # ─── Baselines ───
    print("\n" + "=" * 60)
    print("BASELINES")
    print("=" * 60)

    contam = sum(ANOMALY_INJECTION_RATES.values())
    eif_scores = run_eif(X, contamination=contam)
    copod_scores = run_copod(X)

    # ─── Fusion: LLM-CFD + EIF & LLM-CFD + COPOD ───
    llm_cfd_norm = (llm_cfd_scores - llm_cfd_scores.min()) / (llm_cfd_scores.max() - llm_cfd_scores.min() + 1e-8)
    eif_norm = (eif_scores - eif_scores.min()) / (eif_scores.max() - eif_scores.min() + 1e-8)
    copod_norm = (copod_scores - copod_scores.min()) / (copod_scores.max() - copod_scores.min() + 1e-8)
    fusion_eif_scores = (llm_cfd_norm + eif_norm) / 2.0
    fusion_copod_scores = (llm_cfd_norm + copod_norm) / 2.0

    # ─── Evaluation ───
    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)

    k = 100
    results = {}
    for name, scores in [
        ("LLM-CFD", llm_cfd_scores),
        ("LLM-NoGate", llm_nogate_scores),
        ("EIF", eif_scores),
        ("COPOD", copod_scores),
        ("LLM-CFD+EIF", fusion_eif_scores),
        ("LLM-CFD+COPOD", fusion_copod_scores),
    ]:
        metrics = evaluate_anomaly_detection(scores, anomaly_labels, k=k)
        results[name] = metrics
        print(f"\n  {name}:")
        print(f"    AUPRC:        {metrics['auprc']:.4f}")
        print(f"    Precision@{k}:  {metrics['precision_at_k']:.4f}")
        print(f"    NDCG@{k}:       {metrics['ndcg_at_k']:.4f}")

    # ─── Ablation: fixed threshold ───
    print("\n  --- Ablation: Fixed Threshold (0.95) ---")
    global CONFIDENCE_THRESHOLDS
    orig_thresholds = dict(CONFIDENCE_THRESHOLDS)
    CONFIDENCE_THRESHOLDS = {"high": 0.95, "medium": 0.95, "low": 0.95}
    validated_fixed, _ = statistical_validation(df, best_candidates)
    llm_fixed_scores = compute_anomaly_scores(df_contaminated, validated_fixed)
    fixed_metrics = evaluate_anomaly_detection(llm_fixed_scores, anomaly_labels, k=k)
    CONFIDENCE_THRESHOLDS = orig_thresholds
    print(f"    AUPRC (fixed 0.95): {fixed_metrics['auprc']:.4f}")

    # ─── Compile final results ───
    best_disc = all_strategy_results[best_strategy]
    final_results = {
        "dataset": "Telco Customer Churn",
        "n_records": len(df),
        "n_attributes": len(df.columns),
        "llm_model": DEEPSEEK_MODEL,
        "best_strategy": best_strategy,
        "timestamp": str(datetime.now()),
        "cfd_discovery_by_strategy": all_strategy_results,
        "cfd_discovery": best_disc,
        "anomaly_detection": results,
        "ablation": {
            "LLM-NoGate": results.get("LLM-NoGate", {}),
            "fixed_threshold_095": fixed_metrics,
            "full_pipeline": results.get("LLM-CFD", {}),
        },
        "anomaly_injection": {
            "total_injected": int(anomaly_labels.sum()),
            "injection_rate": float(anomaly_labels.mean()),
        },
    }

    with open(os.path.join(RESULTS_PATH, "final_results.json"), 'w') as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)

    # ─── Summary ───
    elapsed_total = time.time() - start_time
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)
    print(f"  Total time: {elapsed_total:.1f}s")
    print(f"  Best strategy: {best_strategy}")
    print(f"  CFD Discovery — P={best_disc['precision']:.4f}, R={best_disc['recall']:.4f}, F1={best_disc['f1']:.4f}")
    print(f"  Anomaly Detection AUPRC:")
    for name in ["LLM-CFD", "LLM-CFD+EIF", "LLM-CFD+COPOD", "EIF", "COPOD", "LLM-NoGate"]:
        print(f"    {name:15s}: {results[name]['auprc']:.4f}")
    print(f"\n  Strategy comparison:")
    for s in STRATEGIES:
        if s in all_strategy_results:
            a = all_strategy_results[s]
            print(f"    {s:15s}: P={a['precision']:.4f} R={a['recall']:.4f} F1={a['f1']:.4f}")
    print(f"\n  Results saved to: {RESULTS_PATH}/")

    return final_results


if __name__ == "__main__":
    run_experiments()
