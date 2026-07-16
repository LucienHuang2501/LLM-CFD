# LLM-CFD Experiment Data

## Overview

This directory contains all datasets, ground-truth rules, and configuration files
for the LLM-CFD experiments on the Telco Customer Churn and UCI Adult datasets.

## Datasets

### 1. Telco Customer Churn (`telco_churn.csv`)

- **Source**: Kaggle (IBM Sample Dataset)
- **DOI**: [10.34740/KAGGLE/DS/235667](https://doi.org/10.34740/KAGGLE/DS/235667)
- **Licence**: CC0 (Public Domain)
- **Accessed**: 2024-12-01
- **Size**: 7,043 records × 21 attributes (20 features + 1 label)
- **Preprocessing**:
  - Removed `customerID` column (identifier, not a feature)
  - Converted `TotalCharges` from string to numeric (coerce errors to 0)
  - Converted `SeniorCitizen` from int (0/1) to string ("0"/"1") for categorical consistency
- **Attribute Groups** (see paper Table 2):
  - Demographic: gender, SeniorCitizen, Partner, Dependents
  - Service: PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies
  - Account: tenure, Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges
  - Label: Churn

### 2. UCI Adult (`census+income/`)

- **Source**: UCI Machine Learning Repository
- **DOI**: [10.24432/C5GP7S](https://doi.org/10.24432/C5GP7S)
- **Licence**: UCI ML Repository Standard Licence
- **Accessed**: 2024-12-01
- **Size**: 32,561 records × 15 attributes (14 features + 1 label)
- **Preprocessing**: Removed rows with missing values (`?`), selected 10 core attributes, subsampled to 10,000 records (seed=42)
- **Files**: `adult.data` (training), `adult.test` (test), `adult.names` (attribute description), `Index` (metadata)

## Ground-Truth CFD Rules

### `gt_rules_telco.json`

15 manually annotated CFD rules covering 5 constraint types:
- `fd` (function dependency): 4 rules
- `enum` (enumeration): 5 rules
- `range` (numerical range): 3 rules
- `logic` (logical constraint): 1 rule (tenure ≥ 0)
- `consistency` (cross-column computation): 1 rule (TotalCharges ≈ tenure × MonthlyCharges)

Each rule contains: `type`, `condition_attributes`, `condition_values`,
`dependent_attribute`, `expected_pattern`.

### `gt_rules_adult.json`

12 manually annotated CFD rules for the UCI Adult dataset.

## Anomaly Injection

Anomalies are injected into the clean dataset using a fixed random seed
(`RANDOM_SEED=42`) for full reproducibility. The injection scheme follows
5 patterns (see paper Table 4):

| Pattern | Description | Ratio |
|---|---|---|
| Type violation | Change value to violate enum/range constraint | 30% |
| Value swap | Swap values between semantically related columns | 25% |
| Range violation | Set numerical value outside valid range | 20% |
| Consistency break | Break cross-column computation (TotalCharges ≠ tenure × MonthlyCharges) | 15% |
| Logic violation | Set tenure to negative value | 10% |

- Total anomaly ratio: ~17% (1,191 / 7,043 records)
- Each anomaly is designed to violate at least one ground-truth CFD rule

## Configuration

- **Random seed**: `RANDOM_SEED=42` (defined in `src/self_consistency_voting_v2.py`)
- **LLM API models**:
  - DeepSeek: `deepseek-chat` (primary experiments)
  - Zhipu: `glm-4.7-flash` (cross-LLM validation)
- **API response caching**: All LLM API responses are cached as JSON files in
  `results/supplementary/cache_*/` directories to enable replication without
  additional API calls. Each cache directory contains a `metadata.json` with
  model version, timestamp, and call parameters.

## File Listing

```
data/
├── README.md                  # This file
├── telco_churn.csv            # Telco Churn dataset (preprocessed)
├── census+income/             # UCI Adult dataset (raw files)
│   ├── adult.data             # Training data (32,561 rows)
│   ├── adult.test             # Test data
│   ├── adult.names            # Attribute description
│   └── Index                  # Metadata
├── gt_rules_telco.json        # 15 ground-truth CFD rules for Telco
└── gt_rules_adult.json        # 12 ground-truth CFD rules for Adult
```

## Reproduction

```bash
# Install dependencies
pip install -r ../requirements.txt

# Set API key
export DEEPSEEK_API_KEY="your-api-key-here"

# Run main experiment (LLM-CFD discovery + anomaly detection)
cd ..
python3 src/unified_experiment.py

# Run supplementary experiments (CFDMiner, CTANE, fusion)
python3 src/supplement_experiments.py
```

For experiments using cached API responses (no API key needed):
- Cached responses are in `results/supplementary/cache_cot/` (CoT+少样本 15 runs)
- Main experiment caches in `results/cache/`
- Cross-LLM caches in `results/cross_llm/cache/`
- Temperature ablation caches in `results/hscv_temperature/cache/`

## Licence

- Datasets: See respective source licences (CC0 for Telco, UCI Standard for Adult)
- Ground-truth rules and configuration files: CC-BY 4.0
- Source code: MIT Licence
