# LLM-CFD 实验套件

LLM 驱动的条件函数依赖（Conditional Functional Dependency, CFD）发现与异常检测实验代码。基于 Kaggle Telco Customer Churn 数据集，使用 DeepSeek-V4-Pro 大语言模型完成 CFD 规则的语义发现、统计验证、异常评分与可解释输出全流程。

## 目录结构

```
experiments/
├── data/
│   └── telco_churn.csv              # 数据集（Kaggle Telco Customer Churn，7,043 行 × 21 列）
├── src/
│   ├── experiment.py                # 主实验脚本：Phase 1-4 完整流水线
│   └── supplementary_experiments.py # 补充实验脚本：消融、稳定性、统计基线
├── results/
│   ├── cache/                       # LLM 响应缓存（避免重复调用）
│   ├── supplementary/               # 补充实验结果
│   ├── candidate_cfds.json          # 候选 CFD 规则
│   ├── validated_cfds.json          # 通过统计验证的 CFD 规则
│   ├── validation_stats.json        # Phase 2 验证统计
│   └── final_results.json           # 最终实验结果（含所有指标）
├── run_experiment.sh                # 实验运行入口脚本
└── README.md
```

## 运行环境

- Python ≥ 3.10
- 依赖包：`openai`, `pandas`, `numpy`, `scikit-learn`, `scipy`
- DeepSeek API Key（环境变量 `DEEPSEEK_API_KEY`）

安装依赖：

```bash
pip install openai pandas numpy scikit-learn scipy
```

## 快速开始

```bash
# 1. 设置 DeepSeek API Key
export DEEPSEEK_API_KEY="sk-your-key-here"

# 2. 运行主实验
cd experiments
bash run_experiment.sh
# 或直接调用：
python3 src/experiment.py

# 3. 运行补充实验（消融 / 稳定性 / 统计基线）
python3 src/supplementary_experiments.py
```

> 提示：LLM 响应会缓存到 `results/cache/`，再次运行时不会重复消耗 API 调用。如需强制重新调用 LLM，请删除对应缓存文件。

## 代码用途

### 1. 主实验 `src/experiment.py`

完整实现 LLM-CFD 四阶段流水线：

| 阶段 | 功能 | 关键函数 |
|------|------|----------|
| Phase 1 | LLM 语义 CFD 发现（Zero-Shot / Few-Shot / CoT+Few-Shot 三种提示策略） | `build_llm_prompt()`, `call_deepseek_api()`, `parse_cfd_response()` |
| Phase 2 | 统计验证门（支持度过滤 + 自适应置信度门 + Jaccard 去重） | `statistical_validation()`, `deduplicate_cfds()` |
| Phase 3 | 异常评分（构建 n×m 违规矩阵 V，加权聚合，99% 分位裁剪归一化） | `compute_anomaly_scores()` |
| Phase 4 | 可解释输出（top-5% 记录 + 自然语言解释） | 嵌入主流程 |

基线方法：`run_eif()`（Extended Isolation Forest）、`run_copod()`（COPOD Copula 离群点检测）。

评估指标：CFD 发现使用 Precision / Recall / F1（按 `dependent_attribute + condition_attributes + 类型类别` 匹配）；异常检测使用 AUPRC / Precision@k / NDCG@k。

### 2. 补充实验 `src/supplementary_experiments.py`

针对审稿意见的三组补充实验：

- **Experiment 1 — 列名语义消融**：将列名替换为 `ATTR_01`、`ATTR_02`… 测试 LLM 是否依赖列名语义。函数：`experiment_column_ablation()`
- **Experiment 2 — LLM 输出稳定性**：重复调用 LLM 5 次（`STABILITY_RUNS`），统计候选数 / Precision / Recall / F1 的均值与标准差。函数：`experiment_stability()`
- **Experiment 3 — 统计 CFD 基线**：纯统计方法发现 CFD（枚举约束、范围约束、分组 FD、一致性相关性），不调用 LLM。函数：`experiment_statistical_baseline()`

## 核心代码说明

### 提示词模板

主实验 `experiment.py` 中的提示词由 `build_llm_prompt(schema_metadata, df_sample, strategy)` 构建，支持三种策略：

- **zero_shot**：仅含 Schema 描述 + 任务说明 + 输出格式要求
- **few_shot**：在 zero_shot 基础上追加 10 条 CFD 示例（`CFD_EXAMPLES_FEWSHOT` 常量，涵盖 range / enum / logic / consistency / fd 五类）
- **cot_fewshot**（默认）：在 few_shot 基础上追加 7 步 CoT 推理引导（级联服务依赖 → 合同数值范围 → 人口枚举 → 业务逻辑 → 数值一致性 → 全局范围 → 渠道枚举）

输出格式要求 LLM 严格返回 JSON 数组，每条 CFD 含 `type / condition_attributes / condition_values / dependent_attribute / expected_pattern / confidence_estimate / natural_language_description` 七个字段。

补充实验 `supplementary_experiments.py` 使用简化版提示词 `build_prompt(schema, strategy)`，便于在列名消融场景下复用。

### 规则验证脚本

Phase 2 统计验证在 `experiment.py` 中实现：

- `evaluate_cfd_condition(df, cfd)`：按 `condition_values` 过滤行，返回条件满足的布尔掩码
- `evaluate_cfd_dependency(df, cfd, condition_mask)`：在条件满足的行内检查依赖属性是否满足 `expected_pattern`
  - `range` 类型：数值落在 `[min, max]` 区间
  - `enum` 类型：取值属于允许集合
  - `fd / logic / consistency` 类型：非空即视为满足（简化处理）
- `statistical_validation(df, candidates)`：逐条计算 support 与 confidence，应用 `SUPPORT_THRESHOLD=0.01` 与自适应置信度门（high=0.90 / medium=0.85 / low=0.75），输出通过 / 拒绝标签
- `deduplicate_cfds(cfds, jaccard_threshold=0.8)`：按条件属性集 Jaccard 相似度去重，保留置信度更高者

补充实验 `supplementary_experiments.py` 中的 `validate_cfd(df, cfd)` 是等价的独立实现，返回 `{cfd, support, confidence, passed}`。

### 评分计算脚本

Phase 3 异常评分在 `experiment.py` 的 `compute_anomaly_scores(df, validated_cfds)` 中实现：

1. 构建 n×m 违规矩阵 V（n=记录数，m=验证通过的 CFD 数）
   - `range`：归一化超出量 `max((lo-v)/range, (v-hi)/range, 0)`
   - `enum`：违规为 1，否则 0
   - `fd / logic / consistency`：二值违规
2. 按各 CFD 的 confidence 加权求和并归一化：`scores = Σ(V · conf) / Σ(conf)`
3. 99% 分位裁剪后归一化到 [0, 1]

异常检测评估在 `evaluate_anomaly_detection(scores, labels, k=100)` 中计算 AUPRC（`sklearn.metrics.average_precision_score`）、Precision@k、NDCG@k（`sklearn.metrics.ndcg_score`）。

合成异常注入在 `inject_anomalies(df)` 中实现，三类注入：
- 依赖破坏（10%）：破坏 Contract → MonthlyCharges 关系
- 范围违规（5%）：将数值推到 3.5σ~5.5σ 之外
- 逻辑矛盾（3%）：篡改 InternetService ↔ OnlineSecurity 级联

## 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `SUPPORT_THRESHOLD` | 0.01 | Phase 2 支持度下限 θ_s |
| `CONFIDENCE_THRESHOLDS` | high=0.90 / medium=0.85 / low=0.75 | 自适应置信度门 |
| `ANOMALY_INJECTION_RATES` | dep=0.10 / range=0.05 / logic=0.03 | 合成异常注入比例 |
| `RANDOM_SEED` | 42 | 全局随机种子 |
| `LLM_TEMPERATURE` | 0.0 | LLM 采样温度（确定性输出） |
| `LLM_MAX_TOKENS` | 4096 | LLM 最大输出 token |
| `STABILITY_RUNS` | 5 | 补充实验 2 重复调用次数 |

## 主要结果

主实验（`results/final_results.json`）：

- 最佳策略：`few_shot`（F1=0.711，Precision=0.640，Recall=0.800）
- 异常检测 AUPRC：
  - LLM-CFD（本方法）：0.566
  - LLM-CFD + COPOD：0.583
  - LLM-CFD + EIF：0.543
  - COPOD：0.363 / EIF：0.456 / LLM-NoGate：0.504

补充实验（`results/supplementary/supplementary_results.json`）：

- 列名消融后 F1=0（验证 LLM 依赖列名语义）
- 5 次稳定性测试：F1 均值 0.154 ± 0.173（存在波动）
- 统计基线作为无 LLM 对照

## 注意事项

- `supplementary_experiments.py` 中的 `DATA_PATH` 与 `RESULTS_PATH` 为绝对路径，迁移环境时需修改。
- 缓存文件名含 prompt 哈希，修改提示词后会自动生成新缓存，不会污染旧结果。
- `build_ground_truth_cfds()` 定义了 22 条领域知识 ground truth CFD，作为评估基准。
