# LLM-CFD 实验套件

LLM 驱动的条件函数依赖（Conditional Functional Dependency, CFD）发现与异常检测实验代码。基于 Kaggle Telco Customer Churn 数据集和 UCI Adult Census Income 数据集，使用 DeepSeek-V4-Flash 大语言模型完成 CFD 规则的语义发现、统计验证、异常评分与可解释输出全流程。

## 目录结构

```
├── experiments/
│   ├── data/
│   │   ├── telco_churn.csv              # Telco Customer Churn（7,043行×21列，CC0）
│   │   ├── census+income/               # UCI Adult Census Income（CC BY 4.0）
│   │   ├── gt_rules_telco.json          # Telco 人工标注 ground-truth CFD（15条）
│   │   └── gt_rules_adult.json          # Adult 人工标注 ground-truth CFD（12条）
│   ├── src/
│   │   ├── experiment.py                # 主实验：Phase 1-4 完整流水线
│   │   ├── supplementary_experiments.py # 补充实验：消融、稳定性、统计基线
│   │   ├── unified_experiment.py        # 统一评估流程（含CTANE对比）
│   │   ├── self_consistency_voting_v2.py # HSCV 混合自洽性投票机制
│   │   ├── adult_hscv_experiment.py     # Adult 数据集 HSCV 实验
│   │   ├── adult_unified_experiment.py  # Adult 统一评估
│   │   ├── cross_llm_experiment.py      # 跨 LLM 模型验证（DeepSeek vs GLM）
│   │   ├── hscv_ablation_temperature.py # 温度消融实验
│   │   ├── ctane_experiment.py          # CTANE 基线对比
│   │   ├── cfdminer_hyperparam_sweep.py # CFDMiner 超参搜索
│   │   ├── non_semantic_anomaly_experiment.py # 非语义异常控制实验
│   │   ├── generate_cot_runs.py         # CoT+少样本 15次确定性运行生成
│   │   ├── regenerate_figures_nature.py # Nature 风格论文图表生成
│   │   └── regenerate_figures_v4.py     # 论文图表生成脚本
│   ├── results/
│   │   ├── cache/                       # LLM 响应缓存（主实验）
│   │   ├── supplementary/               # 补充实验结果（含 cache/ 和 cache_cot/）
│   │   ├── cross_llm/                   # 跨 LLM 实验结果（含 cache/）
│   │   ├── hscv_temperature/            # 温度消融实验结果（含 cache/）
│   │   ├── ctane/                       # CTANE/CFDMiner 实验结果
│   │   ├── unified/                     # 统一评估流程结果
│   │   ├── p1_revision/                 # Adult 数据集 HSCV 结果
│   │   ├── figures/                     # 论文图表（PNG 300DPI + SVG）
│   │   ├── final_results.json           # 主实验最终结果
│   │   ├── low_anomaly_rate_results.json # 5%注入率实验结果
│   │   ├── non_semantic_anomaly_results.json # 非语义控制实验结果
│   │   └── l2_l3_results.json           # L2/L3层风险检验结果
│   ├── PROMPT_TEMPLATES.md              # 完整提示词模板文档
│   ├── requirements.txt                 # Python 依赖
│   ├── run_experiment.sh                # 实验运行入口脚本
│   └── README.md
├── docs/
│   └── SUPPLEMENTARY_THEORY.md          # 补充理论材料（论文正文引用）
├── LICENSE                              # MIT License
└── README.md                            # 本文件
```

## 运行环境

- Python ≥ 3.10
- 依赖包：见 `experiments/requirements.txt`
- DeepSeek API Key（环境变量 `DEEPSEEK_API_KEY`）
- GLM API Key（环境变量 `GLM_API_KEY`，仅跨 LLM 实验需要）

安装依赖：

```bash
pip install -r experiments/requirements.txt
```

## 快速开始

```bash
# 1. 设置 DeepSeek API Key
export DEEPSEEK_API_KEY="sk-your-key-here"

# 2. 运行主实验（CoT+少样本策略，temp=0.0）
cd experiments
python3 src/experiment.py

# 3. 运行统一评估流程（含 CTANE 对比 + 梯度评分 + 冗余消减）
python3 src/unified_experiment.py

# 4. 生成 CoT+少样本 15次确定性运行
python3 src/generate_cot_runs.py

# 5. 跨 LLM 验证（需要 GLM_API_KEY）
export GLM_API_KEY="your-glm-key"
python3 src/cross_llm_experiment.py

# 6. 生成论文图表
python3 src/regenerate_figures_nature.py
```

> 提示：LLM 响应会缓存到 `results/cache/`、`results/supplementary/cache/`、`results/cross_llm/cache/` 和 `results/hscv_temperature/cache/`，再次运行时不会重复消耗 API 调用。如需强制重新调用 LLM，请删除对应缓存文件。

## 代码用途

### 1. 主实验 `src/experiment.py`

完整实现 LLM-CFD 四阶段流水线：

| 阶段 | 功能 | 关键函数 |
|------|------|----------|
| Phase 1 | LLM 语义 CFD 发现（Zero-Shot / Few-Shot / CoT+Few-Shot 三种提示策略） | `build_llm_prompt()`, `call_deepseek_api()`, `parse_cfd_response()` |
| Phase 2 | 统计验证门（支持度过滤 + 自适应置信度门 + Jaccard 去重） | `statistical_validation()`, `deduplicate_cfds()` |
| Phase 3 | 异常评分（构建 n×m 违规矩阵 V，梯度违规度量，冗余消减，加权聚合，p99裁剪） | `compute_anomaly_scores()` |
| Phase 4 | 可解释输出（top-5% 记录 + 自然语言解释） | 嵌入主流程 |

基线方法：`run_eif()`（Extended Isolation Forest）、`run_copod()`（COPOD Copula 离群点检测）。

### 2. 统一评估流程 `src/unified_experiment.py`

论文 §4 主实验使用的统一评估脚本，包含：
- CTANE/CFDMiner 基线对比
- 梯度违规评分（enum/fd: `1-P(dep|condition)`, range: 归一化距离, consistency: 归一化比率）
- 冗余消减（按 `(dep_attr, condition_attributes)` 分组取最大违规）
- 三种聚合策略（top3_mean / max / mean）+ p99 裁剪

### 3. 补充实验 `src/supplementary_experiments.py`

- **列名语义消融**：将列名替换为 `ATTR_01`… 测试 LLM 是否依赖列名语义
- **LLM 输出稳定性**：少样本策略多次调用（`STABILITY_RUNS=5`），统计波动
- **统计 CFD 基线**：纯统计方法发现 CFD，不调用 LLM

### 4. 其他实验脚本

| 脚本 | 用途 |
|------|------|
| `ctane_experiment.py` | CTANE 基线 + 超参数搜索 |
| `cfdminer_hyperparam_sweep.py` | CFDMiner 超参数搜索 |
| `cross_llm_experiment.py` | DeepSeek-V4-Flash vs GLM-4.7-Flash 跨模型验证 |
| `hscv_ablation_temperature.py` | 温度参数消融实验（temp=0.0/0.3/0.5/0.7，各10次） |
| `non_semantic_anomaly_experiment.py` | 非语义异常控制实验（CFD合法的分布离群点） |
| `generate_cot_runs.py` | CoT+少样本策略 15次确定性运行生成 |
| `adult_hscv_experiment.py` | Adult 数据集 HSCV 实验 |
| `adult_unified_experiment.py` | Adult 数据集统一评估 |
| `regenerate_figures_nature.py` | Nature 风格论文图表（seaborn muted色系，色盲安全） |

## 核心代码说明

### 提示词模板

主实验支持三种提示策略：

- **zero_shot**：仅含 Schema 描述 + 任务说明 + 输出格式要求
- **few_shot**：追加 10 条 CFD 示例（涵盖 range / enum / logic / consistency / fd 五类）
- **cot_fewshot**（推荐默认）：在 few_shot 基础上追加 7 步 CoT 推理引导

> 完整提示词模板详见 `experiments/PROMPT_TEMPLATES.md` 和论文附录B。

输出格式要求 LLM 严格返回 JSON 数组，每条 CFD 含 `type / condition_attributes / condition_values / dependent_attribute / expected_pattern / confidence_estimate / natural_language_description` 七个字段。

### 规则验证

Phase 2 统计验证实现：

- `evaluate_cfd_condition(df, cfd)`：按 `condition_values` 过滤行，返回条件满足的布尔掩码
- `evaluate_cfd_dependency(df, cfd, condition_mask)`：检查依赖属性是否满足 `expected_pattern`
  - `range` 类型：数值落在 `[min, max]` 区间
  - `enum` 类型：取值属于允许集合
  - `fd / logic / consistency` 类型：非空即视为满足
- `statistical_validation(df, candidates)`：逐条计算条件支持度与置信度，应用自适应置信度门
- `deduplicate_cfds(cfds, jaccard_threshold=0.8)`：按条件属性集 Jaccard 相似度去重

### 异常评分

Phase 3 异常评分使用梯度违规度量（非二值标志）：

| 约束类型 | 违规度量 |
|---------|---------|
| range | 归一化超出量 `max((lo-v)/range, (v-hi)/range, 0)` |
| enum/fd | `1 - P(dep=val \| condition)` |
| consistency | 归一化比率 `abs(actual - expected) / expected` |
| logic | `1 - P(A \| condition pattern)` |

冗余消减：按 `(dep_attr, condition_attributes)` 元组分组，取组内最大违规值。

### 异常注入

合成异常注入在 `inject_anomalies(df)` 中实现，三类注入：
- 依赖破坏（10%）：破坏 Contract → MonthlyCharges 关系
- 范围违规（5%）：将数值推到 3.5σ~5.5σ 之外
- 逻辑矛盾（3%）：篡改 InternetService ↔ OnlineSecurity 级联

非语义控制实验（`non_semantic_anomaly_experiment.py`）注入三类CFD合法的分布离群点：
- 范围内分布异常点（经clip限制在CFD合法范围内）
- 罕见但合法的分类属性交换（自由属性）
- CFD范围内的相关性偏移

## 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `SUPPORT_THRESHOLD` | 0.01 | Phase 2 支持度下限 θ_s |
| `CONFIDENCE_THRESHOLDS` | high=0.90 / medium=0.85 / low=0.75 | 自适应置信度门 |
| `ANOMALY_INJECTION_RATES` | dep=0.10 / range=0.05 / logic=0.03 | 合成异常注入比例 |
| `RANDOM_SEED` | 42 | 全局随机种子 |
| `LLM_TEMPERATURE` | 0.0 | LLM 采样温度（推荐配置：CoT+少样本, temp=0.0） |
| `LLM_MAX_TOKENS` | 4096 | LLM 最大输出 token |
| `STABILITY_RUNS` | 5 | 补充实验少样本稳定性调用次数 |
| `COT_RUNS` | 15 | CoT+少样本策略确定性运行次数 |

## 主要结果

### 主实验（CoT+少样本策略, temp=0.0, 15次运行）

| 指标 | LLM-CFD | CTANE | CFDMiner | EIF | COPOD |
|------|---------|-------|----------|-----|-------|
| F1 | 0.737 | 0.028 | 0.003 | — | — |
| AUPRC | 0.495 | 0.272 | 0.387 | 0.205 | 0.167 |
| 查全率 | 100% | 53.3% | 26.7% | — | — |

- 在 fd/enum 约束上 LLM-CFD 与 CTANE 查全率持平（100%），性能差异来自 range/logic/consistency 约束
- bootstrap CI (B=1000): LLM-CFD [0.470, 0.521] vs EIF [0.192, 0.221]，p<0.001
- 确定性输出：15/15次运行 IQR=0.000

### LLM-CFDMiner 融合

| 配置 | AUPRC | 额外API调用 |
|------|-------|------------|
| LLM-CFD 单独 | 0.495 | 0 |
| LLM-CFDMiner（评分平均） | 0.561 | 0 |
| LLM-CFDMiner（规则过滤） | 0.560 | +155 |
| LLM-CFD+COPOD 融合 | 0.309 | +1 |

### 跨 LLM 验证

| 模型 | F1 | AUPRC | Jaccard |
|------|-----|-------|---------|
| DeepSeek-V4-Flash | 0.737 | 0.495 | — |
| GLM-4.7-Flash | 0.718 | 0.495 | 0.920 |

### 5% 注入率实验

| 方法 | AUPRC | F1 |
|------|-------|-----|
| LLM-CFD | 0.285 | 0.292 |
| EIF | 0.069 | 0.085 |
| COPOD | 0.049 | 0.034 |

## Ground Truth 标注

人工标注的 ground-truth CFD 规则存储于：

- `data/gt_rules_telco.json`：Telco 数据集 15 条规则（覆盖 fd / range / enum / logic / consistency 五类）
- `data/gt_rules_adult.json`：Adult 数据集 12 条规则

标注者间一致性（Cohen's kappa）：规则存在性维度 kappa=0.82，规则类型维度 kappa=0.75。

## 补充材料

论文正文引用的补充理论材料位于 `docs/SUPPLEMENTARY_THEORY.md`，包含：

1. 双向验证机制详细说明
2. CFD规则冲突消解近似保证（命题2证明）
3. 多层级数据质量风险检验（L1/L2/L3完整结果）
4. 形式化框架完整定义（定义3/3a，命题1/2）
5. CTANE超参数搜索完整结果（Telco + Adult）
6. CoT+少样本策略15次运行详细数据
7. 提示词模板引用
8. 真实世界数据质量验证

## 注意事项

- `supplementary_experiments.py` 中的 `DATA_PATH` 与 `RESULTS_PATH` 为绝对路径，迁移环境时需修改。
- 缓存文件名含 prompt 哈希，修改提示词后会自动生成新缓存，不会污染旧结果。
- Adult 数据集原始文件位于 `data/census+income/` 目录下，实验中自动预处理（去除缺失值、子采样至 10,000 条，seed=42）。
- LLM 模型：DeepSeek API (`model="deepseek-chat"`)，智谱 API (`model="glm-4.7-flash"`)
- 代码仓库地址：https://github.com/LucienHuang2501/LLM-CFD （MIT License）
